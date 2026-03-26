# Changelog

## [1.0.0] - 2026-03-26

First public release.

### Added
- CAALM package and `caalm` CLI for end-to-end CAZyme prediction from FASTA input.
- Three-level hierarchical prediction pipeline:
  - Level 0: binary CAZy / non-CAZy classification
  - Level 1: multi-label major class classification (`GT`, `GH`, `CBM`, `CE`, `PL`, `AA`)
  - Level 2: family-level retrieval using FAISS nearest-neighbour search
- Packaged installation via `pyproject.toml`, exported package version, and `caalm --version` / `caalm -v`.
- Output files: `*_predictions.tsv`, `*_probabilities.jsonl`, and `*_statistics.tsv`.
- Optional per-level embedding export via `--save-level0-embeddings`, `--save-level1-embeddings`, and `--save-level2-embeddings`.
- Automatic model download from Hugging Face (`lczong/CAALM`) when local paths are not provided
- Mixed precision support (`--mixed-precision bf16|fp16|fp32`)
- Project metadata and release assets including `CITATION.cff`.
