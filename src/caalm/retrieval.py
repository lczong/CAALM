import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import Level2Result

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


@dataclass(frozen=True)
class NeighborHit:
    family: str
    label: str
    ref_sequence_id: str
    score: float
    rank: int


class ProjectionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        init_logit_scale: float = 10.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_labels, output_dim))
        self.prototype_bias = nn.Parameter(torch.zeros(num_labels))
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(init_logit_scale), dtype=torch.float32)
        )
        nn.init.normal_(self.prototypes, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return F.normalize(z, p=2, dim=1)


def require_faiss() -> None:
    if faiss is None:
        raise ImportError(
            "faiss is required for level2 retrieval but is not installed in this Python environment."
        )


def normalize_families(families: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for family in families:
        family_name = str(family).strip().upper()
        if not family_name or family_name in seen:
            continue
        normalized.append(family_name)
        seen.add(family_name)
    return normalized


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def infer_model_dims(ckpt: dict) -> Tuple[int, int]:
    state_dict = ckpt["model_state_dict"]
    hidden_dim = state_dict["encoder.0.weight"].shape[0]
    output_dim = state_dict["encoder.4.weight"].shape[0]
    return hidden_dim, output_dim


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> ProjectionClassifier:
    ckpt_args = ckpt.get("args", {})
    hidden_dim, output_dim = infer_model_dims(ckpt)
    model = ProjectionClassifier(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt_args.get("hidden_dim", hidden_dim)),
        output_dim=int(ckpt_args.get("output_dim", output_dim)),
        num_labels=int(ckpt["num_labels"]),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        init_logit_scale=float(ckpt_args.get("init_logit_scale", 10.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def choose_device(device_name: Optional[str], ckpt: dict) -> torch.device:
    requested = device_name or ckpt.get("args", {}).get("device", "cuda")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def project_embeddings(
    model: ProjectionClassifier,
    embeddings: np.ndarray,
    expected_input_dim: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    input_array = np.asarray(embeddings, dtype=np.float32)
    if input_array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape {input_array.shape}.")
    if input_array.shape[0] == 0:
        raise ValueError("No embeddings were provided for level2 projection.")
    if input_array.shape[1] != expected_input_dim:
        raise ValueError(
            f"Input embedding dimension mismatch: got {input_array.shape[1]}, "
            f"expected {expected_input_dim} from checkpoint."
        )

    projected_batches: List[np.ndarray] = []
    tensor_embeddings = torch.from_numpy(input_array)
    with torch.no_grad():
        for start in range(0, len(tensor_embeddings), batch_size):
            end = start + batch_size
            batch = tensor_embeddings[start:end].to(device, non_blocking=True)
            projected = model.encode(batch).cpu().numpy().astype(np.float32, copy=False)
            projected_batches.append(projected)

    return np.ascontiguousarray(np.concatenate(projected_batches, axis=0))


def save_projected_embeddings(
    seq_ids: Sequence[str],
    projected: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for seq_id, row in zip(seq_ids, projected):
            writer.writerow([seq_id, *row.tolist()])


def load_family_references(
    families: Sequence[str],
    faiss_dir: Path,
    label_tsv_dir: Path,
    id_column: str,
    label_column: str,
) -> Dict[str, dict]:
    require_faiss()

    references: Dict[str, dict] = {}
    for family_name in normalize_families(families):
        index_path = faiss_dir / f"{family_name}.faiss"
        label_candidates = [
            label_tsv_dir / f"{family_name}_trainval.tsv",
            label_tsv_dir / f"{family_name}_labels.tsv",
        ]
        label_path = next((path for path in label_candidates if path.exists()), None)

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if label_path is None:
            expected_paths = ", ".join(str(path) for path in label_candidates)
            raise FileNotFoundError(f"Missing label TSV. Checked: {expected_paths}")

        index = faiss.read_index(str(index_path))
        labels: List[str] = []
        sequence_ids: List[str] = []
        with open(label_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"Missing TSV header in {label_path}")
            if id_column not in reader.fieldnames:
                raise ValueError(f"Missing column '{id_column}' in {label_path}")
            if label_column not in reader.fieldnames:
                raise ValueError(f"Missing column '{label_column}' in {label_path}")

            for row in reader:
                sequence_ids.append(str(row[id_column]))
                labels.append(str(row[label_column]))

        if len(labels) != index.ntotal:
            raise ValueError(
                f"Reference size mismatch for {family_name}: {label_path} has {len(labels)} rows "
                f"but {index_path} has {index.ntotal} vectors."
            )

        references[family_name] = {
            "index": index,
            "labels": labels,
            "sequence_ids": sequence_ids,
        }

    return references


def gather_neighbor_hits(
    references: Dict[str, dict],
    projected: np.ndarray,
    k: int,
    candidate_families: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, List[List[NeighborHit]]]:
    require_faiss()
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    n_queries = projected.shape[0]
    queries = np.ascontiguousarray(projected, dtype=np.float32)

    # Pre-normalize candidate families per query for fast lookup.
    normalized_candidates: Optional[List[set]] = None
    if candidate_families is not None:
        normalized_candidates = [
            {str(f).strip().upper() for f in cf if str(f).strip()}
            for cf in candidate_families
        ]

    family_hits: Dict[str, List[List[NeighborHit]]] = {}
    for family, ref in references.items():
        # Determine which queries need this family.
        if normalized_candidates is not None:
            query_indices = [i for i in range(n_queries) if family in normalized_candidates[i]]
        else:
            query_indices = list(range(n_queries))

        # Initialize all slots with empty lists.
        per_query_hits: List[List[NeighborHit]] = [[] for _ in range(n_queries)]

        if query_indices:
            index = ref["index"]
            labels = ref["labels"]
            ref_ids = ref["sequence_ids"]
            search_k = min(int(k), index.ntotal)

            # Search only the subset of queries that need this family.
            subset_queries = queries[query_indices]
            scores, indices = index.search(subset_queries, search_k)

            # Scatter results back to original positions.
            for sub_idx, orig_idx in enumerate(query_indices):
                hits: List[NeighborHit] = []
                for rank, (score, neighbor_idx) in enumerate(
                    zip(scores[sub_idx], indices[sub_idx]), start=1
                ):
                    if neighbor_idx < 0:
                        continue
                    hits.append(
                        NeighborHit(
                            family=family,
                            label=labels[neighbor_idx],
                            ref_sequence_id=ref_ids[neighbor_idx],
                            score=float(score),
                            rank=rank,
                        )
                    )
                per_query_hits[orig_idx] = hits

        family_hits[family] = per_query_hits

    return family_hits


def pick_consensus_label(
    hits: Sequence[NeighborHit],
) -> Tuple[Optional[str], Optional[NeighborHit], int]:
    if not hits:
        return None, None, 0

    by_label: Dict[str, List[NeighborHit]] = defaultdict(list)
    for hit in hits:
        by_label[hit.label].append(hit)

    ranked = sorted(
        by_label.items(),
        key=lambda item: (
            -len(item[1]),
            -max(hit.score for hit in item[1]),
            -sum(hit.score for hit in item[1]) / len(item[1]),
            item[0],
        ),
    )
    best_label, label_hits = ranked[0]
    best_hit = max(label_hits, key=lambda hit: hit.score)
    return best_label, best_hit, len(label_hits)


def build_prediction_rows(
    seq_ids: Sequence[str],
    families: Sequence[str],
    family_hits: Dict[str, List[List[NeighborHit]]],
    candidate_families: Optional[Sequence[Sequence[str]]] = None,
) -> Tuple[List[dict], List[str]]:
    normalized_families = normalize_families(families)
    if candidate_families is not None and len(candidate_families) != len(seq_ids):
        raise ValueError("candidate_families length must match seq_ids length.")

    rows: List[dict] = []
    for row_idx, seq_id in enumerate(seq_ids):
        allowed_families = (
            normalize_families(candidate_families[row_idx])
            if candidate_families is not None
            else normalized_families
        )
        allowed_family_set = set(allowed_families)
        per_major_class: Dict[str, dict] = {}
        row = {
            "sequence_id": seq_id,
            "candidate_families": "|".join(allowed_families) if allowed_families else None,
        }

        for family in normalized_families:
            hits = family_hits[family][row_idx]
            if family in allowed_family_set:
                family_label, family_best_hit, family_vote_count = pick_consensus_label(hits)
            else:
                family_label, family_best_hit, family_vote_count = None, None, 0

            row[f"{family}_label"] = family_label
            row[f"{family}_score"] = (
                np.nan if family_best_hit is None else family_best_hit.score
            )
            row[f"{family}_match_sequence_id"] = (
                None if family_best_hit is None else family_best_hit.ref_sequence_id
            )
            row[f"{family}_vote_count"] = family_vote_count
            per_major_class[family] = {
                "predicted_family": family_label,
                "score": (
                    None if family_best_hit is None else float(family_best_hit.score)
                ),
                "match_sequence_id": (
                    None if family_best_hit is None else family_best_hit.ref_sequence_id
                ),
                "vote_count": family_vote_count,
            }

        row["per_major_class"] = per_major_class
        rows.append(row)

    ordered_cols = [
        "sequence_id",
        "candidate_families",
    ]
    for family in normalized_families:
        ordered_cols.extend(
            [
                f"{family}_label",
                f"{family}_score",
                f"{family}_match_sequence_id",
                f"{family}_vote_count",
            ]
        )

    return rows, ordered_cols


def run_level2(
    seq_ids: Sequence[str],
    embeddings: np.ndarray,
    checkpoint_path: Path | str,
    families: Optional[Sequence[str]],
    faiss_dir: Path | str,
    label_tsv_dir: Path | str,
    candidate_families: Optional[Sequence[Sequence[str]]] = None,
    label_column: str = "label",
    id_column: str = "sequence_id",
    k: int = 3,
    batch_size: int = 512,
    device_name: Optional[str] = None,
    level1_classes: Optional[Sequence[str]] = None,
) -> Level2Result:
    checkpoint_path = Path(checkpoint_path)
    faiss_dir = Path(faiss_dir)
    label_tsv_dir = Path(label_tsv_dir)

    # When the user did not override families, infer the minimal set from
    # candidate_families so we only load the FAISS indices that are actually
    # needed (e.g. 2 instead of all 6).
    if families is None:
        if candidate_families is not None:
            needed: set[str] = set()
            for cf in candidate_families:
                needed.update(str(f).strip().upper() for f in cf if str(f).strip())
            families = [f for f in (level1_classes or []) if f in needed]
        if not families:
            families = list(level1_classes or [])

    normalized_families = normalize_families(families or [])
    if not normalized_families:
        raise ValueError("At least one family is required for level2 prediction.")
    if len(seq_ids) != len(embeddings):
        raise ValueError(
            f"Level2 prediction requires the same number of ids and embeddings, "
            f"got {len(seq_ids)} ids and {len(embeddings)} embeddings."
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing level2 checkpoint: {checkpoint_path}")

    bootstrap_device = torch.device("cpu")
    ckpt = load_checkpoint(checkpoint_path, bootstrap_device)
    device = choose_device(device_name, ckpt)
    model = build_model_from_checkpoint(ckpt, device)

    projected = project_embeddings(
        model=model,
        embeddings=embeddings,
        expected_input_dim=int(ckpt["input_dim"]),
        device=device,
        batch_size=batch_size,
    )
    references = load_family_references(
        families=normalized_families,
        faiss_dir=faiss_dir,
        label_tsv_dir=label_tsv_dir,
        id_column=id_column,
        label_column=label_column,
    )
    family_hits = gather_neighbor_hits(
        references=references, projected=projected, k=k,
        candidate_families=candidate_families,
    )
    rows, columns = build_prediction_rows(
        seq_ids=seq_ids,
        families=normalized_families,
        family_hits=family_hits,
        candidate_families=candidate_families,
    )

    return Level2Result(
        ids=list(seq_ids),
        families=normalized_families,
        projected_embeddings=projected,
        rows=rows,
        columns=columns,
        device=str(device),
    )


run_level2_prediction = run_level2
