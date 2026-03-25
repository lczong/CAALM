from __future__ import annotations

from typing import Optional

from .classifier import SequenceClassifier
from .io import (
    load_sequences_from_fasta,
    write_level1_embeddings,
    write_prediction_outputs,
    write_statistics,
)
from .retrieval import run_retrieval
from .types import PredictionResult


class PredictionPipeline:
    def __init__(self, device: Optional[str] = None, mixed_precision: str = "bf16"):
        self.classifier = SequenceClassifier(
            device=device,
            mixed_precision=mixed_precision,
        )
        self.device = self.classifier.device
        self.dtype = self.classifier.dtype
        self.mixed_precision = self.classifier.mixed_precision
        self.level1_classes = self.classifier.level1_classes

        self.all_sequences: dict[str, str] = {}
        self.all_ids: list[str] = []
        self.level0_results = None
        self.level1_results = None
        self.retrieval_results = None
        self.positive_ids: set[str] = set()

    def predict(
        self,
        test_fasta: str,
        level0_model_path: Optional[str] = None,
        level1_model_path: Optional[str] = None,
        level0_threshold: float = 0.5,
        level1_thresholds: Optional[list[float]] = None,
        level1_thresholds_file: Optional[str] = None,
        level1_global_threshold: float = 0.5,
        level2_model_path: str = "./models/level2/model.pt",
        level2_families: Optional[list[str]] = None,
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
        dataloader_workers: int = 4,
    ) -> PredictionResult:
        records = load_sequences_from_fasta(test_fasta)
        sequences = [record.sequence for record in records]
        ids = [record.sequence_id for record in records]
        self.all_sequences = {record.sequence_id: record.sequence for record in records}
        self.all_ids = ids
        self.level0_results = None
        self.level1_results = None
        self.retrieval_results = None
        self.positive_ids = set()

        print(f"\n{'='*60}")
        print("LEVEL 0: Classification (CAZy vs Non-CAZy)")
        print(f"{'='*60}")

        self.classifier.load_level0_model(level0_model_path)

        print(f"\nRunning on {len(sequences)} sequences from {test_fasta}")
        level0_results = self.classifier.predict_level0(
            sequences=sequences,
            ids=ids,
            batch_size=batch_size,
            max_length=max_length,
            threshold=level0_threshold,
            save_embeddings=False,
            dataloader_workers=dataloader_workers,
        )
        self.level0_results = level0_results
        self.positive_ids = set(level0_results.positive_ids)
        self.classifier.unload_model()

        level1_results = None
        retrieval_results = None
        positive_records = [
            record
            for record, is_positive in zip(records, level0_results.positive_mask)
            if bool(is_positive)
        ]

        if positive_records:
            positive_sequences = [record.sequence for record in positive_records]
            positive_ids_list = [record.sequence_id for record in positive_records]

            print(f"\n{'='*60}")
            print("LEVEL 1: Classification (GT, GH, CBM, CE, PL, AA)")
            print(f"{'='*60}")

            self.classifier.load_level1_model(level1_model_path)

            print(f"\nRunning on {len(positive_sequences)} positive sequences")
            level1_results = self.classifier.predict_level1(
                sequences=positive_sequences,
                ids=positive_ids_list,
                batch_size=batch_size,
                max_length=max_length,
                thresholds=level1_thresholds,
                thresholds_file=level1_thresholds_file,
                global_threshold=level1_global_threshold,
                save_embeddings=True,
                dataloader_workers=dataloader_workers,
            )
            self.level1_results = level1_results
            self.classifier.unload_model()

            if level1_results is not None:
                candidate_families = (
                    [list(level2_families) for _ in level1_results.ids]
                    if level2_families
                    else level1_results.predicted_labels
                )

                print(f"\n{'='*60}")
                print("LEVEL 2: Retrieval Prediction")
                print(f"{'='*60}")

                retrieval_results = run_retrieval(
                    seq_ids=level1_results.ids,
                    embeddings=level1_results.embeddings,
                    checkpoint_path=level2_model_path,
                    families=level2_families,
                    faiss_dir=level2_faiss_dir,
                    label_tsv_dir=level2_label_tsv_dir,
                    candidate_families=candidate_families,
                    label_column=level2_label_column,
                    id_column=level2_id_column,
                    k=level2_k,
                    batch_size=level2_batch_size,
                    device_name=level2_device,
                    level1_classes=self.level1_classes,
                )
                self.retrieval_results = retrieval_results

                assigned_count = sum(
                    1
                    for row in retrieval_results.rows
                    if any(
                        details.get("predicted_family")
                        for details in row.get("per_major_class", {}).values()
                    )
                )
                print("\nLevel 2 Prediction Results:")
                print(f"   Total sequences: {len(level1_results.ids)}")
                print(
                    f"   Assigned labels: {assigned_count} "
                    f"({assigned_count/len(level1_results.ids)*100:.2f}%)"
                )
                for family in retrieval_results.families:
                    family_count = sum(
                        1
                        for row in retrieval_results.rows
                        if row.get("per_major_class", {}).get(family, {}).get(
                            "predicted_family"
                        )
                    )
                    print(
                        f"   {family}: {family_count} "
                        f"({family_count/len(level1_results.ids)*100:.2f}%)"
                    )

        print("\n" + "=" * 60)
        print("PREDICTION COMPLETE!")
        print("=" * 60)

        write_prediction_outputs(
            level0_results=level0_results,
            level1_results=level1_results,
            retrieval_results=retrieval_results,
            output_dir=output_dir,
            output_name=output_name,
            level1_classes=self.level1_classes,
        )

        if save_embeddings:
            write_level1_embeddings(
                level1_results=level1_results,
                output_dir=output_dir,
                output_name=output_name,
            )

        write_statistics(
            level0_results=level0_results,
            level1_results=level1_results,
            retrieval_results=retrieval_results,
            output_dir=output_dir,
            output_name=output_name,
            level1_classes=self.level1_classes,
        )

        return PredictionResult(
            level0=level0_results,
            level1=level1_results,
            retrieval=retrieval_results,
        )
