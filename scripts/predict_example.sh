#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

caalm \
  --input example/example.fasta \
  --level0-model models/level0 \
  --level1-model models/level1 \
  --level2-model models/level2/model.pt \
  --level2-faiss-dir models/level2/faiss \
  --level2-label-tsv-dir models/level2/refdb \
  --level2-label-column label \
  --output-dir outputs \
  --output-name example \
  "$@"
