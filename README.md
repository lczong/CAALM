# CAALM: Carbohydrate Activity Annotation with protein Language Models

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/lczong/CAALM.git
    cd CAALM
    ```

2.  **Set Up a Virtual Environment (Recommended)**
    ```bash
    conda create -n caalm
    conda activate caalm
    ```

3.  **Install Dependencies**
    ```
    tqdm
    numpy
    biopython
    torch
    transformers
    faiss-cpu  # or faiss-gpu for level2 retrieval
    ```

4.  **Download Model Assets**

    Download the full [CAALM](https://huggingface.co/lczong/CAALM) Hugging Face repository and save it locally as a directory named `models` in the project root.

    The expected layout is:

    ```text
    models/
    ├── level0/
    ├── level1/
    └── level2/
        ├── model.pt
        ├── faiss/
        └── refdb/
    ```

## 📖 Usage

### Prediction Flow

CAALM runs three levels in sequence:

1. Level 0 predicts whether a sequence is CAZy or non-CAZy.
2. If Level 0 predicts CAZy, Level 1 predicts one or more major CAZy classes from `GT`, `GH`, `CBM`, `CE`, `PL`, and `AA`.
3. Level 2 retrieves family labels from the FAISS index and reference database for each predicted Level 1 major class.

If Level 1 predicts multiple classes such as `GH|CBM`, Level 2 searches both major-class databases and writes one family prediction per major class.

### Example Command

With the local model assets in this repo:

```bash
python src/predict.py \
  --input example/example.fasta \
  --level0-model models/level0 \
  --level1-model models/level1 \
  --level2-model models/level2/model.pt \
  --level2-faiss-dir models/level2/faiss \
  --level2-label-tsv-dir models/level2/refdb \
  --level2-label-column label \
  --output-dir outputs \
  --output-name example
```

If your environment does not allow multiprocessing dataloader workers, add:

```bash
--num-workers 0
```

### Model Sources

- The recommended setup is to download the full [CAALM](https://huggingface.co/lczong/CAALM) Hugging Face repository into a local directory named `models`.
- Level 0 and Level 1 can also be loaded from local directories via `--level0-model` and `--level1-model`.
- If `--level0-model` or `--level1-model` are omitted, the code will try to download those from Hugging Face automatically.
- Level 2 uses the local retrieval assets under `models/level2`.

### Outputs

Each run writes three main files under `--output-dir` with the prefix `--output-name`.

`*_predictions.tsv`
- `sequence_id`
- `pred_is_cazy`
- `pred_cazy_class`
- `pred_cazy_family`

Notes:
- `pred_is_cazy` is `1` for CAZy and `0` for non-CAZy.
- `pred_cazy_class` is empty for non-CAZy sequences.
- `pred_cazy_family` is empty for non-CAZy sequences.
- For multi-label Level 1 predictions, both `pred_cazy_class` and `pred_cazy_family` use `|` as the separator.

`*_probabilities.jsonl`
- One JSON object per sequence.
- `level0.prob_is_cazy`: probability from the binary classifier.
- `level1.class_probabilities`: probabilities for `GT`, `GH`, `CBM`, `CE`, `PL`, and `AA`.
- `level2.predicted_families`: family predictions for each predicted major class, including score, matched reference sequence, and vote count.

`*_statistics.tsv`
- Summary counts and percentages for Level 0, Level 1, and Level 2 outputs.

### Level 2 Inputs

Level 2 expects:
- a checkpoint from `--level2-model`
- per-major-class FAISS indices in `--level2-faiss-dir`
- per-major-class reference TSVs in `--level2-label-tsv-dir`

The current repo layout uses:
- `models/level2/model.pt`
- `models/level2/faiss/<CLASS>.faiss`
- `models/level2/refdb/<CLASS>_labels.tsv`

If `--level2-families` is omitted, Level 2 automatically uses each sequence's predicted Level 1 classes.
