# CAALM: Carbohydrate Activity Annotation with protein Language Models

[![PyPI](https://img.shields.io/pypi/v/caalm)](https://pypi.org/project/caalm/)

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/lczong/CAALM.git
    cd CAALM
    ```

2.  **Set Up a Virtual Environment (Recommended)**
    ```bash
    conda create -n caalm python=3.10
    conda activate caalm
    ```

3.  **Install PyTorch**

    Follow the installation below, or choose the build that matches your device ([official guide](https://pytorch.org/get-started/locally/) | [previous versions](https://pytorch.org/get-started/previous-versions/))

    ```bash
    # CUDA 12.6
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

    # CPU only
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    ```

4.  **Install FAISS**

    ```bash
    # CPU (via pip or conda)
    pip install faiss-cpu        # option 1
    conda install faiss-cpu -c pytorch  # option 2

    # GPU (conda recommended — pip may not work correctly)
    conda install faiss-gpu -c pytorch
    ```

5.  **Install the Package**

    ```bash
    # From pip
    pip install caalm # option 1
    
    # From this repository
    pip install . # option 2
    ```

6.  **Download Model Assets**

    Download the full [CAALM](https://huggingface.co/lczong/CAALM) Hugging Face repository into a directory named `models` in the project root:

    ```bash
    python -c "from huggingface_hub import snapshot_download; snapshot_download('lczong/CAALM', local_dir='models')"
    ```

    The expected layout after download is:

    ```text
    models/
    ├── level0/          # Level 0 binary classifier
    ├── level1/          # Level 1 multi-label classifier
    └── level2/
        ├── model.pt     # Level 2 projection checkpoint
        ├── faiss/       # FAISS indices (<CLASS>.faiss)
        └── refdb/       # Reference TSVs (<CLASS>_labels.tsv)
    ```

## 📖 Usage

### Prediction Flow

CAALM runs three levels in sequence:

1. Level 0 predicts whether a sequence is `CAZy` or `non-CAZy`.
2. If Level 0 predicts CAZy, Level 1 predicts one or more major CAZy classes from `GT`, `GH`, `CBM`, `CE`, `PL`, and `AA`.
3. Level 2 retrieves family labels from the FAISS index and reference database for each predicted Level 1 major class.

If Level 1 predicts multiple classes such as `GH|CBM`, Level 2 searches both major-class databases and writes one family prediction per major class.

### Example Command

A convenience script is provided to run the example with one command:

```bash
./scripts/predict_example.sh
```

Or invoke the CLI directly:

```bash
caalm input/example.fasta
```

The output name defaults to the input filename stem (here `example`, from `input/example.fasta`), and output files are written to `./outputs/`. To customise:

```bash
caalm your_sequences.fasta -o results --output-name my_run
```

Use `caalm --help` to see all options grouped by category.

### Common Options

```bash
# Use a specific GPU
caalm input.fasta -d cuda:0

# Enable mixed precision for faster inference
caalm input.fasta --mixed-precision bf16

# Increase batch size for large-memory GPUs
caalm input.fasta -b 16

# Increase the level 2 projection batch size independently
caalm input.fasta -b2 1024

# Save level 1 embeddings for downstream analysis
caalm input.fasta --save-level1-embeddings

# Save level 0 embeddings
caalm input.fasta --save-level0-embeddings

# Save level 2 projected embeddings
caalm input.fasta --save-level2-embeddings
```

### Models

The recommended setup is to download the full [CAALM](https://huggingface.co/lczong/CAALM) Hugging Face repository into a local `models` directory (see Installation step 6). If local files are not found, Level 0 and Level 1 will try to download from Hugging Face automatically.

| Level | Description | Default path | CLI override |
|-------|-------------|-------------|--------------|
| Level 0 | Binary CAZy / non-CAZy classifier | `./models/level0` | `--level0-model` |
| Level 1 | Multi-label major class classifier | `./models/level1` | `--level1-model` |
| Level 2 | Projection checkpoint | `./models/level2/model.pt` | `--level2-model` |
| Level 2 | FAISS indices (`<CLASS>.faiss`) | `./models/level2/faiss` | `--level2-faiss-dir` |
| Level 2 | Reference TSVs (`<CLASS>_labels.tsv`) | `./models/level2/refdb` | `--level2-label-tsv-dir` |

If `--level2-families` is omitted, Level 2 automatically uses each sequence's predicted Level 1 classes.

### Outputs

Each run writes three main files under `--output-dir` with the prefix `--output-name`. When requested, embedding arrays are also saved as `.npy` files only.

`*_predictions.tsv`
- `sequence_id`
- `pred_is_cazy`
- `pred_cazy_class`
- `pred_cazy_family`

Notes:
- `pred_is_cazy` is `CAZy` for CAZy sequences and `Non-CAZy` for non-CAZy sequences.
- `pred_cazy_class` is empty for non-CAZy sequences.
- `pred_cazy_family` is empty for non-CAZy sequences.
- For multi-label Level 1 predictions, both `pred_cazy_class` and `pred_cazy_family` use `|` as the separator.

`*_probabilities.jsonl`
- One JSON object per sequence.
- `level0.prob_is_cazy`: probability from the binary classifier.
- `level1.class_probabilities`: probabilities for `GT`, `GH`, `CBM`, `CE`, `PL`, and `AA`.
- `level2.predicted_families`: family predictions for each predicted major class, including score, matched reference sequence, and vote count.
- Saved probabilities and Level 2 scores are rounded to 5 decimal places.

`*_statistics.tsv`
- Summary counts and percentages for Level 0, Level 1, and Level 2 outputs.

Optional embedding outputs
- `*_level0_embeddings.npy` when `--save-level0-embeddings` is used.
- `*_level1_embeddings.npy` when `--save-level1-embeddings` is used.
- `*_level2_embeddings.npy` when `--save-level2-embeddings` is used.
