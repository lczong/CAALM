import csv
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from Bio import SeqIO

from .types import Level0Result, Level1Result, Level2Result, SequenceRecord


def load_sequences_from_fasta(fasta_file: str) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        records.append(
            SequenceRecord(
                sequence_id=record.id,
                sequence=str(record.seq).replace(".", ""),
            )
        )
    return records


def round_nested_floats(value: object, digits: int = 5) -> object:
    if isinstance(value, dict):
        return {key: round_nested_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested_floats(item, digits) for item in value]
    if isinstance(value, np.floating):
        return round(float(value), digits)
    if isinstance(value, float):
        return round(value, digits)
    return value


def build_result_maps(
    level0_results: Level0Result,
    level1_results: Optional[Level1Result],
    level2_results: Optional[Level2Result],
    level1_classes: list[str],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    level0_map = {}
    for i, seq_id in enumerate(level0_results.ids):
        level0_map[seq_id] = {
            "pred_is_cazy": level0_results.predicted_labels[i] == "cazy",
            "prob_is_cazy": float(level0_results.probabilities[i, 1]),
        }

    level1_map = {}
    if level1_results is not None:
        for i, seq_id in enumerate(level1_results.ids):
            probs = level1_results.probabilities[i]
            level1_map[seq_id] = {
                "predicted_classes": list(level1_results.predicted_labels[i]),
                "class_probabilities": {
                    class_name: float(probs[j])
                    for j, class_name in enumerate(level1_classes)
                },
            }

    level2_map = {}
    if level2_results is not None:
        for row in level2_results.rows:
            candidate_major_classes = row.get("candidate_families")
            per_major_class = row.get("per_major_class", {})
            predicted_families = []
            split_classes = (
                candidate_major_classes.split("|") if candidate_major_classes else []
            )
            for major_class in split_classes:
                major_class_result = per_major_class.get(major_class, {})
                if major_class_result.get("predicted_family"):
                    predicted_families.append(
                        {
                            "major_class": major_class,
                            "family_label": major_class_result["predicted_family"],
                            "score": major_class_result.get("score"),
                            "match_sequence_id": major_class_result.get(
                                "match_sequence_id"
                            ),
                            "vote_count": major_class_result.get("vote_count"),
                        }
                    )

            level2_map[row["sequence_id"]] = {
                "predicted_families": predicted_families,
                "candidate_major_classes": split_classes,
            }

    return level0_map, level1_map, level2_map


def write_prediction_outputs(
    level0_results: Level0Result,
    level1_results: Optional[Level1Result],
    level2_results: Optional[Level2Result],
    output_dir: str,
    output_name: str,
    level1_classes: list[str],
    *,
    _precomputed_maps: Optional[tuple[dict[str, dict], dict[str, dict], dict[str, dict]]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if _precomputed_maps is not None:
        level0_map, level1_map, level2_map = _precomputed_maps
    else:
        level0_map, level1_map, level2_map = build_result_maps(
            level0_results=level0_results,
            level1_results=level1_results,
            level2_results=level2_results,
            level1_classes=level1_classes,
        )

    predictions_path = Path(output_dir) / f"{output_name}_predictions.tsv"
    with open(predictions_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "sequence_id",
                "pred_is_cazy",
                "pred_cazy_class",
                "pred_cazy_family",
            ]
        )

        for seq_id in level0_results.ids:
            level0_row = level0_map[seq_id]
            level1_row = level1_map.get(seq_id)
            level2_row = level2_map.get(seq_id)
            writer.writerow(
                [
                    seq_id,
                    "CAZy" if level0_row["pred_is_cazy"] else "Non-CAZy",
                    "|".join(level1_row["predicted_classes"]) if level1_row else "",
                    (
                        ""
                        if level2_row is None
                        else "|".join(
                            item["family_label"]
                            for item in level2_row["predicted_families"]
                        )
                    ),
                ]
            )

    print(f"Saved predictions to {predictions_path}")

    probabilities_path = Path(output_dir) / f"{output_name}_probabilities.jsonl"
    with open(probabilities_path, "w") as f:
        for seq_id in level0_results.ids:
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
                    "predicted_classes": (
                        [] if level1_row is None else level1_row["predicted_classes"]
                    ),
                    "class_probabilities": (
                        {class_name: None for class_name in level1_classes}
                        if level1_row is None
                        else level1_row["class_probabilities"]
                    ),
                },
                "level2": {
                    "evaluated": level2_row is not None,
                    "candidate_major_classes": (
                        [] if level1_row is None else level1_row["predicted_classes"]
                    )
                    if level2_row is None
                    else level2_row["candidate_major_classes"],
                    "predicted_families": (
                        [] if level2_row is None else level2_row["predicted_families"]
                    ),
                },
            }
            f.write(json.dumps(round_nested_floats(record)) + "\n")

    print(f"Saved probabilities to {probabilities_path}")


def write_statistics(
    level0_results: Level0Result,
    level1_results: Optional[Level1Result],
    level2_results: Optional[Level2Result],
    output_dir: str,
    output_name: str,
    level1_classes: list[str],
    *,
    _precomputed_maps: Optional[tuple[dict[str, dict], dict[str, dict], dict[str, dict]]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if _precomputed_maps is not None:
        level0_map, level1_map, level2_map = _precomputed_maps
    else:
        level0_map, level1_map, level2_map = build_result_maps(
            level0_results=level0_results,
            level1_results=level1_results,
            level2_results=level2_results,
            level1_classes=level1_classes,
        )

    stats_path = Path(output_dir) / f"{output_name}_statistics.tsv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["level", "category", "count", "denominator", "percentage"])

        total_sequences = len(level0_results.ids)
        cazy_count = sum(1 for row in level0_map.values() if row["pred_is_cazy"])
        writer.writerow(
            [
                "level0",
                "cazy",
                cazy_count,
                total_sequences,
                f"{(100.0 * cazy_count / max(total_sequences, 1)):.2f}",
            ]
        )
        writer.writerow(
            [
                "level0",
                "non_cazy",
                total_sequences - cazy_count,
                total_sequences,
                f"{(100.0 * (total_sequences - cazy_count) / max(total_sequences, 1)):.2f}",
            ]
        )

        level1_denominator = len(level1_map)
        if level1_denominator > 0:
            class_counts = {class_name: 0 for class_name in level1_classes}
            for row in level1_map.values():
                for class_name in row["predicted_classes"]:
                    class_counts[class_name] += 1
            for class_name in level1_classes:
                count = class_counts[class_name]
                writer.writerow(
                    [
                        "level1",
                        class_name,
                        count,
                        level1_denominator,
                        f"{(100.0 * count / level1_denominator):.2f}",
                    ]
                )

        level2_denominator = len(level2_map)
        if level2_denominator > 0:
            assigned_sequences = sum(
                1 for row in level2_map.values() if row["predicted_families"]
            )
            writer.writerow(
                [
                    "level2",
                    "sequences_with_family_prediction",
                    assigned_sequences,
                    level2_denominator,
                    f"{(100.0 * assigned_sequences / level2_denominator):.2f}",
                ]
            )

            major_class_counts = {class_name: 0 for class_name in level1_classes}
            family_counts: dict[str, int] = {}
            for row in level2_map.values():
                for family_result in row["predicted_families"]:
                    major_class_counts[family_result["major_class"]] += 1
                    family_label = family_result["family_label"]
                    family_counts[family_label] = family_counts.get(family_label, 0) + 1

            for class_name in level1_classes:
                count = major_class_counts[class_name]
                if count > 0:
                    writer.writerow(
                        [
                            "level2",
                            class_name,
                            count,
                            level2_denominator,
                            f"{(100.0 * count / level2_denominator):.2f}",
                        ]
                    )

            for family_label in sorted(family_counts):
                count = family_counts[family_label]
                writer.writerow(
                    [
                        "level2_family",
                        family_label,
                        count,
                        level2_denominator,
                        f"{(100.0 * count / level2_denominator):.2f}",
                    ]
                )

    print(f"Saved statistics to {stats_path}")


def write_level1_embeddings(
    level1_results: Optional[Level1Result],
    output_dir: str,
    output_name: str,
) -> None:
    if level1_results is None or level1_results.embeddings is None:
        return

    emb_path = Path(output_dir) / f"{output_name}_level1_embeddings.npy"
    np.save(emb_path, level1_results.embeddings)
    print(f"   Saved Level 1 embeddings to {emb_path}")

    emb_csv_path = Path(output_dir) / f"{output_name}_level1_embeddings.csv"
    with open(emb_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for seq_id, emb in zip(level1_results.ids, level1_results.embeddings):
            writer.writerow([seq_id, *emb.tolist()])
    print(f"   Saved Level 1 embeddings CSV to {emb_csv_path}")
