from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SequenceRecord:
    sequence_id: str
    sequence: str


@dataclass
class Level0Result:
    ids: list[str]
    probabilities: np.ndarray
    predicted_labels: list[str]
    positive_ids: set[str]
    positive_mask: np.ndarray
    embeddings: Optional[np.ndarray]
    threshold: float


@dataclass
class Level1Result:
    ids: list[str]
    probabilities: np.ndarray
    predictions: np.ndarray
    predicted_labels: list[list[str]]
    embeddings: Optional[np.ndarray]
    thresholds: np.ndarray


@dataclass
class Level2Result:
    ids: list[str]
    families: list[str]
    projected_embeddings: np.ndarray
    rows: list[dict]
    columns: list[str]
    device: str


@dataclass
class PredictionResult:
    level0: Level0Result
    level1: Optional[Level1Result]
    level2: Optional[Level2Result]
