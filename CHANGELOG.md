# Changelog

## [1.0.0] - 2026-03-18

First public release.

### Added
- Three-level hierarchical CAZyme prediction pipeline:
  - Level 0: binary CAZy / non-CAZy classification
  - Level 1: multi-label major class classification (GT, GH, CBM, CE, PL, AA)
  - Level 2: family-level retrieval using FAISS nearest-neighbour search
- `caalm` CLI entry point with full argument support
- `caalm --version` / `caalm -v` flag
- Output files: `*_predictions.tsv`, `*_probabilities.jsonl`, `*_statistics.tsv`
- Optional Level 1 embedding export (`--save-embeddings`)
- Mixed precision support (`--mixed-precision bf16|fp16|fp32`)
- Per-class and per-file threshold configuration for Level 1
- Automatic model download from Hugging Face (`lczong/CAALM`) when local paths are not provided
- `pyproject.toml` for pip installation with `faiss-cpu` or `faiss-gpu` extras
