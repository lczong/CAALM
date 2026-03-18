import os
import json
import csv
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from tqdm import tqdm
from Bio import SeqIO
from .level2 import run_level2_prediction


class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences: List[str], ids: List[str],
                 tokenizer, max_length: int = 1024):
        self.sequences = sequences
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }


class CAALMPredictor:
    """
    Three-level protein sequence predictor combining level0, level1, and level2 prediction.
    Level 0: classification (cazy vs non-cazy)
    Level 1: classification for positive samples
    Level 2: retrieval prediction for level1-positive samples
    """

    def __init__(self, device: str = None, mixed_precision: str = 'bf16'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if mixed_precision == 'bf16' else torch.float16 if mixed_precision == 'fp16' else torch.float32
        self.mixed_precision = mixed_precision
        print(f"Device: {self.device}, Mixed Precision: {mixed_precision}")

        self.level0_model = None
        self.level1_model = None
        self.tokenizer = None
        self.data_collator = None

        self.all_sequences = {}
        self.all_ids = []

        self.level0_results = None
        self.level1_results = None
        self.level2_results = None
        self.positive_ids = set()

        self.level0_label_to_id = {'non-cazy': 0, 'cazy': 1}
        self.level0_id_to_label = {0: 'non-cazy', 1: 'cazy'}

        self.level1_classes = ["GT", "GH", "CBM", "CE", "PL", "AA"]

    def _load_default_model(self, preferred_subfolder: str, legacy_subfolder: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("lczong/CAALM", subfolder=preferred_subfolder)
            model = AutoModelForSequenceClassification.from_pretrained(
                "lczong/CAALM",
                subfolder=preferred_subfolder,
                torch_dtype=self.dtype
            )
            return model
        except Exception as exc:
            print(
                f"Unable to load HuggingFace subfolder '{preferred_subfolder}', "
                f"falling back to legacy subfolder '{legacy_subfolder}': {exc}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("lczong/CAALM", subfolder=legacy_subfolder)
            return AutoModelForSequenceClassification.from_pretrained(
                "lczong/CAALM",
                subfolder=legacy_subfolder,
                torch_dtype=self.dtype
            )

    def load_level0_model(self, model_path: str):
        if model_path:
            print(f"🔬 Loading Level 0 Classification Model from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Level 0 model path not found: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.level0_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=self.dtype
            )
        else:
            print("No local level0 model path provided, will download from HuggingFace")
            self.level0_model = self._load_default_model("level0", "binary")
        self.level0_model.to(self.device)
        self.level0_model.eval()
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        print("   Level 0 model loaded successfully")

    def load_level1_model(self, model_path: str):
        if model_path:
            print(f"🔬 Loading Level 1 Classification Model from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Level 1 model path not found: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.level1_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=self.dtype
            )
        else:
            print("No local level1 model path provided, will download from HuggingFace")
            self.level1_model = self._load_default_model("level1", "multi-label")
        self.level1_model.to(self.device)
        self.level1_model.eval()
        if self.data_collator is None:
            self.data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

        print("   Level 1 model loaded successfully")

    def load_sequences_from_fasta(self, fasta_file: str) -> Tuple[List[str], List[str]]:
        sequences = []
        ids = []

        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_id = record.id
            sequence = str(record.seq).replace('.', '')

            sequences.append(sequence)
            ids.append(seq_id)
            self.all_sequences[seq_id] = sequence

        self.all_ids = ids

        return sequences, ids

    @torch.no_grad()
    def inference(
        self,
        dataset: Dataset,
        model: torch.nn.Module,
        data_collator,
        batch_size: int = 8,
        save_embeddings: bool = False,
        is_level1: bool = False,
        dataloader_workers: int = 4,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            collate_fn=data_collator,
            pin_memory=(self.device == "cuda"),
        )

        model.eval()

        all_probs = []
        all_embs = [] if save_embeddings else None

        if self.mixed_precision == 'bf16' and self.device == "cuda" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif self.mixed_precision == 'fp16' and self.device == "cuda" and torch.cuda.is_fp16_supported():
            autocast_dtype = torch.float16
        else:
            autocast_dtype = None

        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            ctx = torch.amp.autocast(device_type=self.device, dtype=autocast_dtype) if autocast_dtype else torch.amp.autocast(device_type=self.device, enabled=False)
            with ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=save_embeddings,
                    return_dict=True,
                )
                logits = outputs.logits

                if is_level1:
                    probs = logits.float().sigmoid()
                else:
                    probs = logits.float().softmax(dim=-1)

                all_probs.append(probs.detach().cpu())

                if save_embeddings:
                    hidden_states = outputs.hidden_states
                    emb = hidden_states[-1][:, 0, :]
                    all_embs.append(emb.detach().cpu())

        probabilities = torch.cat(all_probs, dim=0).numpy()
        embeddings = torch.cat(all_embs, dim=0).numpy() if save_embeddings else None

        return probabilities, embeddings

    def predict_level0(
        self,
        sequences: List[str],
        ids: List[str],
        batch_size: int = 8,
        max_length: int = 1024,
        threshold: float = 0.95,
        save_embeddings: bool = False,
        dataloader_workers: int = 4,
    ) -> Dict:
        if self.level0_model is None:
            raise RuntimeError("Level 0 model not loaded. Call load_level0_model() first.")

        dataset = ProteinSequenceDataset(
            sequences=sequences,
            ids=ids,
            tokenizer=self.tokenizer,
            max_length=max_length
        )

        probabilities, embeddings = self.inference(
            dataset=dataset,
            model=self.level0_model,
            data_collator=self.data_collator,
            batch_size=batch_size,
            save_embeddings=save_embeddings,
            is_level1=False,
            dataloader_workers=dataloader_workers,
        )

        positive_mask = probabilities[:, 1] > threshold
        positive_ids = set([ids[i] for i in range(len(ids)) if positive_mask[i]])
        predicted_labels = [self.level0_id_to_label[1 if mask else 0] for mask in positive_mask]

        print("\nLevel 0 Classification Results:")
        print(f"   Total sequences: {len(ids)}")
        print(f"   Positive (CAZy): {len(positive_ids)} ({len(positive_ids)/len(ids)*100:.2f}%)")
        print(f"   Negative (Non-CAZy): {len(ids) - len(positive_ids)} ({(len(ids) - len(positive_ids))/len(ids)*100:.2f}%)")

        return {
            'probabilities': probabilities,
            'predicted_labels': predicted_labels,
            'positive_ids': positive_ids,
            'embeddings': embeddings,
            'ids': ids,
            'threshold': threshold
        }

    def predict_level1(
        self,
        sequences: List[str],
        ids: List[str],
        batch_size: int = 8,
        max_length: int = 1024,
        thresholds: Optional[Sequence[float]] = None,
        thresholds_file: Optional[str] = None,
        global_threshold: float = 0.5,
        save_embeddings: bool = False,
        dataloader_workers: int = 4,
    ) -> Dict:
        if self.level1_model is None:
            raise RuntimeError("Level 1 model not loaded. Call load_level1_model() first.")

        if len(sequences) == 0:
            print("\n⚠️ No sequences provided for Level 1")
            return None

        dataset = ProteinSequenceDataset(
            sequences=sequences,
            ids=ids,
            tokenizer=self.tokenizer,
            max_length=max_length
        )

        probabilities, embeddings = self.inference(
            dataset=dataset,
            model=self.level1_model,
            data_collator=self.data_collator,
            batch_size=batch_size,
            save_embeddings=save_embeddings,
            is_level1=True,
            dataloader_workers=dataloader_workers,
        )

        per_class_thr = self.load_thresholds(
            classes=self.level1_classes,
            global_threshold=global_threshold,
            thresholds_list=thresholds,
            thresholds_file=thresholds_file
        )

        predictions = (probabilities >= per_class_thr[None, :]).astype(int)

        predicted_label_lists = []
        for i in range(len(ids)):
            labels = [self.level1_classes[j] for j in range(len(self.level1_classes)) if predictions[i, j] == 1]
            predicted_label_lists.append(labels)

        class_counts = predictions.sum(axis=0)
        print("\nLevel 1 Classification Results:")
        for j, class_name in enumerate(self.level1_classes):
            print(f"   {class_name}: {int(class_counts[j])} ({class_counts[j]/len(ids)*100:.2f}%)")

        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'predicted_labels': predicted_label_lists,
            'embeddings': embeddings,
            'ids': ids,
            'thresholds': per_class_thr
        }

    def load_thresholds(
        self,
        classes: List[str],
        global_threshold: float,
        thresholds_list: Optional[Sequence[float]] = None,
        thresholds_file: Optional[str] = None
    ) -> np.ndarray:
        if thresholds_file:
            with open(thresholds_file, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                thr = np.array([float(obj.get(c, global_threshold)) for c in classes], dtype=np.float32)
            elif isinstance(obj, list):
                if len(obj) != len(classes):
                    raise ValueError(f"thresholds_file list length {len(obj)} != num classes {len(classes)}")
                thr = np.array([float(x) for x in obj], dtype=np.float32)
            else:
                raise ValueError("thresholds_file JSON must be a dict or list")
            return thr

        if thresholds_list is not None:
            if len(thresholds_list) != len(classes):
                raise ValueError(f"thresholds length {len(thresholds_list)} != num classes {len(classes)}")
            return np.array([float(x) for x in thresholds_list], dtype=np.float32)

        return np.full(len(classes), float(global_threshold), dtype=np.float32)

    def predict_level2(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        families: Optional[Sequence[str]] = None,
        candidate_families: Optional[Sequence[Sequence[str]]] = None,
        checkpoint_path: str = "./models/level2/model.pt",
        faiss_dir: str = "./models/level2/faiss",
        label_tsv_dir: str = "./models/level2/refdb",
        label_column: str = "label",
        id_column: str = "sequence_id",
        k: int = 3,
        projection_batch_size: int = 512,
        level2_device: Optional[str] = None,
    ) -> Optional[Dict]:
        if embeddings is None:
            raise RuntimeError("Level 2 prediction requires level1 embeddings.")
        if len(ids) == 0:
            print("\n⚠️ No sequences provided for Level 2")
            return None
        if len(ids) != len(embeddings):
            raise ValueError(
                f"Level 2 prediction requires the same number of ids and embeddings, "
                f"got {len(ids)} ids and {len(embeddings)} embeddings."
            )

        if families is None:
            inferred = set()
            if candidate_families is not None:
                for family_list in candidate_families:
                    for family in family_list:
                        family_name = str(family).strip().upper()
                        if family_name:
                            inferred.add(family_name)
            families = [family for family in self.level1_classes if family in inferred]
        else:
            families = [str(family).strip().upper() for family in families if str(family).strip()]

        if not families:
            print("\n⚠️ No level1 families available for Level 2")
            return None

        results = run_level2_prediction(
            seq_ids=ids,
            embeddings=embeddings,
            checkpoint_path=Path(checkpoint_path),
            families=families,
            faiss_dir=Path(faiss_dir),
            label_tsv_dir=Path(label_tsv_dir),
            candidate_families=candidate_families,
            label_column=label_column,
            id_column=id_column,
            k=k,
            batch_size=projection_batch_size,
            device_name=level2_device,
        )

        assigned_count = sum(
            1
            for row in results["rows"]
            if any(
                details.get("predicted_family")
                for details in row.get("per_major_class", {}).values()
            )
        )
        print("\nLevel 2 Prediction Results:")
        print(f"   Total sequences: {len(ids)}")
        print(f"   Assigned labels: {assigned_count} ({assigned_count/len(ids)*100:.2f}%)")
        for family in results["families"]:
            family_count = sum(
                1
                for row in results["rows"]
                if row.get("per_major_class", {}).get(family, {}).get("predicted_family")
            )
            print(f"   {family}: {family_count} ({family_count/len(ids)*100:.2f}%)")

        return results

    def _build_result_maps(
        self,
        level0_results: Dict,
        level1_results: Optional[Dict],
        level2_results: Optional[Dict],
    ) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
        level0_map = {}
        for i, seq_id in enumerate(level0_results["ids"]):
            level0_map[seq_id] = {
                "pred_is_cazy": level0_results["predicted_labels"][i] == "cazy",
                "prob_is_cazy": float(level0_results["probabilities"][i, 1]),
            }

        level1_map = {}
        if level1_results:
            for i, seq_id in enumerate(level1_results["ids"]):
                probs = level1_results["probabilities"][i]
                level1_map[seq_id] = {
                    "predicted_classes": list(level1_results["predicted_labels"][i]),
                    "class_probabilities": {
                        class_name: float(probs[j])
                        for j, class_name in enumerate(self.level1_classes)
                    },
                }

        level2_map = {}
        if level2_results:
            for row in level2_results["rows"]:
                candidate_major_classes = row.get("candidate_families")
                per_major_class = row.get("per_major_class", {})
                predicted_families = []
                for major_class in (
                    candidate_major_classes.split("|") if candidate_major_classes else []
                ):
                    major_class_result = per_major_class.get(major_class, {})
                    if major_class_result.get("predicted_family"):
                        predicted_families.append(
                            {
                                "major_class": major_class,
                                "family_label": major_class_result["predicted_family"],
                                "score": major_class_result.get("score"),
                                "match_sequence_id": major_class_result.get("match_sequence_id"),
                                "vote_count": major_class_result.get("vote_count"),
                            }
                        )

                level2_map[row["sequence_id"]] = {
                    "predicted_families": predicted_families,
                    "candidate_major_classes": (
                        candidate_major_classes.split("|")
                        if candidate_major_classes
                        else []
                    ),
                }

        return level0_map, level1_map, level2_map

    def save_prediction_outputs(
        self,
        level0_results: Dict,
        level1_results: Optional[Dict],
        level2_results: Optional[Dict],
        output_dir: str,
        output_name: str,
    ):
        os.makedirs(output_dir, exist_ok=True)
        level0_map, level1_map, level2_map = self._build_result_maps(
            level0_results=level0_results,
            level1_results=level1_results,
            level2_results=level2_results,
        )

        predictions_path = Path(output_dir) / f"{output_name}_predictions.tsv"
        with open(predictions_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "sequence_id",
                "pred_is_cazy",
                "pred_cazy_class",
                "pred_cazy_family",
            ])

            for seq_id in level0_results["ids"]:
                level0_row = level0_map[seq_id]
                level1_row = level1_map.get(seq_id)
                level2_row = level2_map.get(seq_id)
                writer.writerow([
                    seq_id,
                    int(level0_row["pred_is_cazy"]),
                    "|".join(level1_row["predicted_classes"]) if level1_row else "",
                    (
                        ""
                        if level2_row is None
                        else "|".join(
                            item["family_label"]
                            for item in level2_row["predicted_families"]
                        )
                    ),
                ])

        print(f"Saved predictions to {predictions_path}")

        probabilities_path = Path(output_dir) / f"{output_name}_probabilities.jsonl"
        with open(probabilities_path, "w") as f:
            for seq_id in level0_results["ids"]:
                level0_row = level0_map[seq_id]
                level1_row = level1_map.get(seq_id)
                level2_row = level2_map.get(seq_id)

                record = {
                    "sequence_id": seq_id,
                    "level0": {
                        "prob_is_cazy": level0_row["prob_is_cazy"],
                    },
                    "level1": {
                        "evaluated": level1_row is not None,
                        "predicted_classes": [] if level1_row is None else level1_row["predicted_classes"],
                        "class_probabilities": (
                            {class_name: None for class_name in self.level1_classes}
                            if level1_row is None
                            else level1_row["class_probabilities"]
                        ),
                    },
                    "level2": {
                        "evaluated": level2_row is not None,
                        "candidate_major_classes": (
                            (
                                [] if level1_row is None else level1_row["predicted_classes"]
                            )
                            if level2_row is None
                            else level2_row["candidate_major_classes"]
                        ),
                        "predicted_families": (
                            [] if level2_row is None else level2_row["predicted_families"]
                        ),
                    },
                }
                f.write(json.dumps(record) + "\n")

        print(f"Saved probabilities to {probabilities_path}")

    def save_statistics(
        self,
        level0_results: Dict,
        level1_results: Optional[Dict],
        level2_results: Optional[Dict],
        output_dir: str,
        output_name: str,
    ):
        os.makedirs(output_dir, exist_ok=True)
        level0_map, level1_map, level2_map = self._build_result_maps(
            level0_results=level0_results,
            level1_results=level1_results,
            level2_results=level2_results,
        )

        stats_path = Path(output_dir) / f"{output_name}_statistics.tsv"
        with open(stats_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["level", "category", "count", "denominator", "percentage"])

            total_sequences = len(level0_results["ids"])
            cazy_count = sum(1 for row in level0_map.values() if row["pred_is_cazy"])
            writer.writerow(["level0", "cazy", cazy_count, total_sequences, f"{(100.0 * cazy_count / max(total_sequences, 1)):.2f}"])
            writer.writerow(["level0", "non_cazy", total_sequences - cazy_count, total_sequences, f"{(100.0 * (total_sequences - cazy_count) / max(total_sequences, 1)):.2f}"])

            level1_denominator = len(level1_map)
            if level1_denominator > 0:
                class_counts = {class_name: 0 for class_name in self.level1_classes}
                for row in level1_map.values():
                    for class_name in row["predicted_classes"]:
                        class_counts[class_name] += 1
                for class_name in self.level1_classes:
                    count = class_counts[class_name]
                    writer.writerow(["level1", class_name, count, level1_denominator, f"{(100.0 * count / level1_denominator):.2f}"])

            level2_denominator = len(level2_map)
            if level2_denominator > 0:
                assigned_sequences = sum(
                    1 for row in level2_map.values() if row["predicted_families"]
                )
                writer.writerow([
                    "level2",
                    "sequences_with_family_prediction",
                    assigned_sequences,
                    level2_denominator,
                    f"{(100.0 * assigned_sequences / level2_denominator):.2f}",
                ])

                major_class_counts = {class_name: 0 for class_name in self.level1_classes}
                family_counts = {}
                for row in level2_map.values():
                    for family_result in row["predicted_families"]:
                        major_class_counts[family_result["major_class"]] += 1
                        family_label = family_result["family_label"]
                        family_counts[family_label] = family_counts.get(family_label, 0) + 1

                for class_name in self.level1_classes:
                    count = major_class_counts[class_name]
                    if count > 0:
                        writer.writerow(["level2", class_name, count, level2_denominator, f"{(100.0 * count / level2_denominator):.2f}"])

                for family_label in sorted(family_counts):
                    count = family_counts[family_label]
                    writer.writerow(["level2_family", family_label, count, level2_denominator, f"{(100.0 * count / level2_denominator):.2f}"])

        print(f"Saved statistics to {stats_path}")

    def save_level1_embeddings(
        self,
        level1_results: Optional[Dict],
        output_dir: str,
        output_name: str,
    ):
        if not level1_results or level1_results["embeddings"] is None:
            return

        emb_path = f"{output_dir}/{output_name}_level1_embeddings.npy"
        np.save(emb_path, level1_results["embeddings"])
        print(f"   Saved Level 1 embeddings to {emb_path}")

        emb_csv_path = f"{output_dir}/{output_name}_level1_embeddings.csv"
        with open(emb_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for seq_id, emb in zip(level1_results["ids"], level1_results["embeddings"]):
                writer.writerow([seq_id, *emb.tolist()])
        print(f"   Saved Level 1 embeddings CSV to {emb_csv_path}")

    def predict(
        self,
        test_fasta: str,
        level0_model_path: Optional[str] = None,
        level1_model_path: Optional[str] = None,
        level0_threshold: float = 0.5,
        level1_thresholds: Optional[List[float]] = None,
        level1_thresholds_file: Optional[str] = None,
        level1_global_threshold: float = 0.5,
        level2_model_path: str = "./models/level2/model.pt",
        level2_families: Optional[List[str]] = None,
        level2_faiss_dir: str = "./models/level2/faiss",
        level2_label_tsv_dir: str = "./models/level2/refdb",
        level2_label_column: str = "label",
        level2_id_column: str = "sequence_id",
        level2_k: int = 3,
        level2_batch_size: int = 512,
        level2_device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 1024,
        output_dir: str = "./outputs",
        output_name: str = "test",
        save_embeddings: bool = False,
        dataloader_workers: int = 4
    ):
        sequences, ids = self.load_sequences_from_fasta(test_fasta)

        level0_results = None
        level1_results = None
        level2_results = None
        need_level1_embeddings = True

        print(f"\n{'='*60}")
        print("LEVEL 0: Classification (CAZy vs Non-CAZy)")
        print(f"{'='*60}")

        self.load_level0_model(level0_model_path)

        print(f"\nRunning on {len(sequences)} sequences from {test_fasta}")

        level0_results = self.predict_level0(
            sequences=sequences,
            ids=ids,
            batch_size=batch_size,
            max_length=max_length,
            threshold=level0_threshold,
            save_embeddings=False,
            dataloader_workers=dataloader_workers,
        )

        self.level0_results = level0_results
        self.positive_ids = level0_results['positive_ids']

        if len(self.positive_ids) > 0:
            positive_sequences = [self.all_sequences[id] for id in self.positive_ids if id in self.all_sequences]
            positive_ids_list = [id for id in self.positive_ids if id in self.all_sequences]

            print(f"\n{'='*60}")
            print("LEVEL 1: Classification (GT, GH, CBM, CE, PL, AA)")
            print(f"{'='*60}")

            self.load_level1_model(level1_model_path)

            print(f"\nRunning on {len(positive_sequences)} positive sequences")

            if len(positive_sequences) > 0:
                level1_results = self.predict_level1(
                    sequences=positive_sequences,
                    ids=positive_ids_list,
                    batch_size=batch_size,
                    max_length=max_length,
                    thresholds=level1_thresholds,
                    thresholds_file=level1_thresholds_file,
                    global_threshold=level1_global_threshold,
                    save_embeddings=need_level1_embeddings,
                    dataloader_workers=dataloader_workers,
                )
                self.level1_results = level1_results

                candidate_families = (
                    [level2_families for _ in level1_results["ids"]]
                    if level2_families
                    else level1_results["predicted_labels"]
                )

                print(f"\n{'='*60}")
                print("LEVEL 2: Retrieval Prediction")
                print(f"{'='*60}")

                level2_results = self.predict_level2(
                    embeddings=level1_results["embeddings"],
                    ids=level1_results["ids"],
                    families=level2_families,
                    candidate_families=candidate_families,
                    checkpoint_path=level2_model_path,
                    faiss_dir=level2_faiss_dir,
                    label_tsv_dir=level2_label_tsv_dir,
                    label_column=level2_label_column,
                    id_column=level2_id_column,
                    k=level2_k,
                    projection_batch_size=level2_batch_size,
                    level2_device=level2_device,
                )
                self.level2_results = level2_results

        print("\n" + "="*60)
        print("PREDICTION COMPLETE!")
        print("="*60)

        self.save_prediction_outputs(
            level0_results=level0_results,
            level1_results=level1_results,
            level2_results=level2_results,
            output_dir=output_dir,
            output_name=output_name,
        )

        if save_embeddings:
            self.save_level1_embeddings(
                level1_results=level1_results,
                output_dir=output_dir,
                output_name=output_name
            )

        self.save_statistics(
            level0_results=level0_results,
            level1_results=level1_results,
            level2_results=level2_results,
            output_dir=output_dir,
            output_name=output_name,
        )
