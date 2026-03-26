import json
import os
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from .types import Level0Result, Level1Result


class ProteinSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        ids: list[str],
        tokenizer,
        max_length: int = 1024,
    ):
        self.sequences = sequences
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        sequence = self.sequences[idx]
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


class SequenceClassifier:
    def __init__(self, device: Optional[str] = None, mixed_precision: str = "bf16"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16
            if mixed_precision == "bf16"
            else torch.float16 if mixed_precision == "fp16" else torch.float32
        )
        self.mixed_precision = mixed_precision
        print(f"Device: {self.device}, Mixed Precision: {mixed_precision}")

        if str(self.device).startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        self.model = None
        self.tokenizer = None
        self.data_collator = None

        self.level0_id_to_label = {0: "non-cazy", 1: "cazy"}
        self.level1_classes = ["GT", "GH", "CBM", "CE", "PL", "AA"]

    def _load_default_model(self, preferred_subfolder: str, legacy_subfolder: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "lczong/CAALM", subfolder=preferred_subfolder
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "lczong/CAALM",
                subfolder=preferred_subfolder,
                dtype=self.dtype,
            )
            return model
        except Exception as exc:
            print(
                f"Unable to load HuggingFace subfolder '{preferred_subfolder}', "
                f"falling back to legacy subfolder '{legacy_subfolder}': {exc}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "lczong/CAALM", subfolder=legacy_subfolder
            )
            return AutoModelForSequenceClassification.from_pretrained(
                "lczong/CAALM",
                subfolder=legacy_subfolder,
                dtype=self.dtype,
            )

    def _finalize_loaded_model(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)

    def load_level0_model(self, model_path: Optional[str]) -> None:
        if model_path:
            print(f"🔬 Loading Level 0 Classification Model from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Level 0 model path not found: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                dtype=self.dtype,
            )
        else:
            print("No local level0 model path provided, will download from HuggingFace")
            model = self._load_default_model("level0", "binary")

        self._finalize_loaded_model(model)
        print("   Level 0 model loaded successfully")

    def load_level1_model(self, model_path: Optional[str]) -> None:
        if model_path:
            print(f"🔬 Loading Level 1 Classification Model from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Level 1 model path not found: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                dtype=self.dtype,
            )
        else:
            print("No local level1 model path provided, will download from HuggingFace")
            model = self._load_default_model("level1", "multi-label")

        self._finalize_loaded_model(model)
        print("   Level 1 model loaded successfully")

    def unload_model(self) -> None:
        if self.model is None:
            return

        del self.model
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def inference(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        save_embeddings: bool = False,
        is_level1: bool = False,
        dataloader_workers: int = 4,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.model is None or self.data_collator is None:
            raise RuntimeError("No model is loaded. Call load_level0_model() or load_level1_model() first.")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            collate_fn=self.data_collator,
            pin_memory=str(self.device).startswith("cuda"),
        )

        all_probs = []
        all_embs = [] if save_embeddings else None

        # Register a forward hook on the base model to capture the last hidden
        # state without requesting all 33 intermediate layers via
        # output_hidden_states=True (which would waste ~660MB GPU RAM per batch).
        _last_hidden: list[torch.Tensor] = []
        _hook = None
        if save_embeddings:
            def _capture_last_hidden(module, args, output):
                _last_hidden.append(output.last_hidden_state)

            _hook = self.model.base_model.register_forward_hook(_capture_last_hidden)

        # Extract device type ("cuda" or "cpu") for autocast — self.device may
        # contain an index like "cuda:0" which autocast does not accept.
        device_type = torch.device(self.device).type

        if (
            self.mixed_precision == "bf16"
            and device_type == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            autocast_dtype = torch.bfloat16
        elif self.mixed_precision == "fp16" and device_type == "cuda":
            autocast_dtype = torch.float16
        else:
            autocast_dtype = None

        ctx = (
            torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.amp.autocast(device_type=device_type, enabled=False)
        )

        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            with ctx:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits = outputs.logits

                if is_level1:
                    probs = logits.float().sigmoid()
                else:
                    probs = logits.float().softmax(dim=-1)

                all_probs.append(probs.detach().cpu())

                if save_embeddings:
                    emb = _last_hidden.pop()[:, 0, :]
                    all_embs.append(emb.detach().cpu())

        if _hook is not None:
            _hook.remove()

        probabilities = torch.cat(all_probs, dim=0).numpy()
        embeddings = torch.cat(all_embs, dim=0).numpy() if save_embeddings else None
        return probabilities, embeddings

    def predict_level0(
        self,
        sequences: list[str],
        ids: list[str],
        batch_size: int = 8,
        max_length: int = 1024,
        threshold: float = 0.95,
        save_embeddings: bool = False,
        dataloader_workers: int = 4,
    ) -> Level0Result:
        if self.model is None:
            raise RuntimeError("Level 0 model not loaded. Call load_level0_model() first.")

        dataset = ProteinSequenceDataset(
            sequences=sequences,
            ids=ids,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        probabilities, embeddings = self.inference(
            dataset=dataset,
            batch_size=batch_size,
            save_embeddings=save_embeddings,
            is_level1=False,
            dataloader_workers=dataloader_workers,
        )

        positive_mask = probabilities[:, 1] > threshold
        positive_ids = {ids[i] for i in range(len(ids)) if positive_mask[i]}
        predicted_labels = [self.level0_id_to_label[1 if mask else 0] for mask in positive_mask]

        print("\nLevel 0 Classification Results:")
        print(f"   Total sequences: {len(ids)}")
        print(f"   Positive (CAZy): {len(positive_ids)} ({len(positive_ids)/len(ids)*100:.2f}%)")
        print(
            f"   Negative (Non-CAZy): {len(ids) - len(positive_ids)} "
            f"({(len(ids) - len(positive_ids))/len(ids)*100:.2f}%)"
        )

        return Level0Result(
            ids=ids,
            probabilities=probabilities,
            predicted_labels=predicted_labels,
            positive_ids=positive_ids,
            positive_mask=positive_mask,
            embeddings=embeddings,
            threshold=threshold,
        )

    def load_thresholds(
        self,
        classes: list[str],
        global_threshold: float,
        thresholds_list: Optional[Sequence[float]] = None,
        thresholds_file: Optional[str] = None,
    ) -> np.ndarray:
        if thresholds_file:
            with open(thresholds_file, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                thr = np.array(
                    [float(obj.get(c, global_threshold)) for c in classes],
                    dtype=np.float32,
                )
            elif isinstance(obj, list):
                if len(obj) != len(classes):
                    raise ValueError(
                        f"thresholds_file list length {len(obj)} != num classes {len(classes)}"
                    )
                thr = np.array([float(x) for x in obj], dtype=np.float32)
            else:
                raise ValueError("thresholds_file JSON must be a dict or list")
            return thr

        if thresholds_list is not None:
            if len(thresholds_list) != len(classes):
                raise ValueError(
                    f"thresholds length {len(thresholds_list)} != num classes {len(classes)}"
                )
            return np.array([float(x) for x in thresholds_list], dtype=np.float32)

        return np.full(len(classes), float(global_threshold), dtype=np.float32)

    def predict_level1(
        self,
        sequences: list[str],
        ids: list[str],
        batch_size: int = 8,
        max_length: int = 1024,
        thresholds: Optional[Sequence[float]] = None,
        thresholds_file: Optional[str] = None,
        global_threshold: float = 0.5,
        save_embeddings: bool = False,
        dataloader_workers: int = 4,
    ) -> Optional[Level1Result]:
        if self.model is None:
            raise RuntimeError("Level 1 model not loaded. Call load_level1_model() first.")

        if len(sequences) == 0:
            print("\n⚠️ No sequences provided for Level 1")
            return None

        dataset = ProteinSequenceDataset(
            sequences=sequences,
            ids=ids,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        probabilities, embeddings = self.inference(
            dataset=dataset,
            batch_size=batch_size,
            save_embeddings=save_embeddings,
            is_level1=True,
            dataloader_workers=dataloader_workers,
        )

        per_class_thr = self.load_thresholds(
            classes=self.level1_classes,
            global_threshold=global_threshold,
            thresholds_list=thresholds,
            thresholds_file=thresholds_file,
        )
        predictions = (probabilities >= per_class_thr[None, :]).astype(int)

        predicted_label_lists = []
        for i in range(len(ids)):
            labels = [
                self.level1_classes[j]
                for j in range(len(self.level1_classes))
                if predictions[i, j] == 1
            ]
            predicted_label_lists.append(labels)

        class_counts = predictions.sum(axis=0)
        print("\nLevel 1 Classification Results:")
        for j, class_name in enumerate(self.level1_classes):
            print(f"   {class_name}: {int(class_counts[j])} ({class_counts[j]/len(ids)*100:.2f}%)")

        return Level1Result(
            ids=ids,
            probabilities=probabilities,
            predictions=predictions,
            predicted_labels=predicted_label_lists,
            embeddings=embeddings,
            thresholds=per_class_thr,
        )
