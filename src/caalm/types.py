from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

import numpy as np


class DictLikeDataclass:
    """Small compatibility layer for older dict-style result access."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> list[str]:
        return [field.name for field in fields(self)]

    def items(self) -> list[tuple[str, Any]]:
        return [(key, getattr(self, key)) for key in self.keys()]

    def values(self) -> list[Any]:
        return [getattr(self, key) for key in self.keys()]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)


@dataclass
class SequenceRecord(DictLikeDataclass):
    sequence_id: str
    sequence: str


@dataclass
class Level0Result(DictLikeDataclass):
    ids: list[str]
    probabilities: np.ndarray
    predicted_labels: list[str]
    positive_ids: set[str]
    positive_mask: np.ndarray
    embeddings: Optional[np.ndarray]
    threshold: float


@dataclass
class Level1Result(DictLikeDataclass):
    ids: list[str]
    probabilities: np.ndarray
    predictions: np.ndarray
    predicted_labels: list[list[str]]
    embeddings: Optional[np.ndarray]
    thresholds: np.ndarray


@dataclass
class RetrievalResult(DictLikeDataclass):
    ids: list[str]
    families: list[str]
    projected_embeddings: np.ndarray
    rows: list[dict]
    columns: list[str]
    device: str


@dataclass
class PredictionResult(DictLikeDataclass):
    level0: Level0Result
    level1: Optional[Level1Result]
    retrieval: Optional[RetrievalResult]
