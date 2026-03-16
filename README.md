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
    ```

## 📖 Usage

### Basic Prediction

This is the simplest way to run a prediction. The command is designed to work out-of-the-box. It will
- Using the default classification thresholds
- Automatically detecting devices (CPU or GPU)
- Downloading the required model weights from the [CAALM](https://huggingface.co/lczong/CAALM) repository on the Hugging Face Hub.

```bash
python src/predict.py --input example/example.fasta 
```
