# CAALM: Carbohydrate Activity Annotation with protein Language Models

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

    Choose the build that matches your hardware:

    ```bash
    # CUDA 12.6
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

    # CPU only
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    ```

4.  **Install FAISS**

    FAISS is recommended to be installed via Conda:

    ```bash
    # GPU
    conda install faiss-gpu=1.13.2 -c pytorch

    # CPU
    conda install faiss-cpu=1.13.2 -c pytorch
    ```

    If you prefer pip for CPU-only FAISS, you can skip this step and use the pip extras below instead.

5.  **Install the Package**

    If you installed FAISS via Conda in the previous step:

    ```bash
    pip install .
    ```

    If you skipped the Conda FAISS step and want CPU-only FAISS via pip:

    ```bash
    pip install ".[cpu]"
    ```

6.  **Download Model Assets**

    Download the full [CAALM](https://huggingface.co/lczong/CAALM) Hugging Face repository into a directory named `models` in the project root:

    ```bash
    pip install huggingface_hub
    python -c "from huggingface_hub import snapshot_download; snapshot_download('lczong/CAALM', local_dir='models')"
    ```

    The expected layout after download is:

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

A convenience script is provided to run the example with one command:

```bash
./scripts/predict_example.sh
```

Or invoke the CLI directly:

```bash
caalm example/example.fasta
```

The output name defaults to the input filename stem (here `example`), and output files are written to `./outputs/`. To customise:

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

# Save level 1 embeddings for downstream analysis
caalm input.fasta --save-embeddings
```

If your environment does not allow multiprocessing dataloader workers, add:

```bash
--num-workers 0
```

On HPC systems with user-level pip installs (`~/.local`), broken packages can leak into the conda environment and cause import errors. Prefix your command with `PYTHONNOUSERSITE=1` to prevent this:

```bash
PYTHONNOUSERSITE=1 caalm ...
```

### Model Sources

- The recommended setup is to download the full [CAALM](https://huggingface.co/lczong/CAALM) Hugging Face repository into a local directory named `models`.
- Model paths default to `./models/level0`, `./models/level1`, and `./models/level2/` but can be overridden with `--level0-model`, `--level1-model`, and `--level2-model`.
- If local model files are not found, Level 0 and Level 1 will try to download from Hugging Face automatically.

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
