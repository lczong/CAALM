import argparse
import os
import sys

from . import __version__
from .pipeline import PredictionPipeline
from .utils import log_gpu_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAALM: Predict CAZymes and CAZyme classes from protein sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)

    # -- Input / Output -------------------------------------------------------
    io_group = parser.add_argument_group("input/output")
    io_group.add_argument(
        "input", nargs="?", default=None,
        help="path to input FASTA file",
    )
    io_group.add_argument(
        "-i", "--input", dest="input_flag", default=None,
        help="path to input FASTA file (alternative to positional)",
    )
    io_group.add_argument(
        "-o", "--output-dir", default="./outputs",
        help="output directory",
    )
    io_group.add_argument(
        "--output-name", default=None,
        help="prefix for output files (default: input filename stem)",
    )
    io_group.add_argument(
        "--save-embeddings", action="store_true",
        help="save level 1 embeddings to npy and csv",
    )

    # -- Model paths ----------------------------------------------------------
    model_group = parser.add_argument_group("model paths")
    model_group.add_argument(
        "--level0-model", default="./models/level0",
        help="path to level 0 classification model",
    )
    model_group.add_argument(
        "--level1-model", default="./models/level1",
        help="path to level 1 classification model",
    )
    model_group.add_argument(
        "--level2-model", default="./models/level2/model.pt",
        help="path to level 2 projection checkpoint",
    )

    # -- Thresholds -----------------------------------------------------------
    thr_group = parser.add_argument_group("thresholds")
    thr_group.add_argument(
        "--level0-threshold", type=float, default=0.5,
        help="threshold for level 0 classification",
    )
    thr_group.add_argument(
        "--level1-threshold", type=float, default=0.5,
        help="global threshold for level 1 classification",
    )
    thr_group.add_argument(
        "--level1-thresholds", type=float, nargs=6,
        help="per-class thresholds: GT GH CBM CE PL AA",
    )
    thr_group.add_argument(
        "--level1-thresholds-file",
        help="JSON file with per-class thresholds",
    )

    # -- Level 2 retrieval ----------------------------------------------------
    l2_group = parser.add_argument_group("level 2 retrieval")
    l2_group.add_argument(
        "--level2-families", nargs="*",
        help="override retrieval families; defaults to level 1 predictions",
    )
    l2_group.add_argument(
        "--level2-faiss-dir", default="./models/level2/faiss",
        help="directory containing <family>.faiss indices",
    )
    l2_group.add_argument(
        "--level2-label-tsv-dir", default="./models/level2/refdb",
        help="directory containing <family>_labels.tsv files",
    )
    l2_group.add_argument(
        "--level2-label-column", default="label",
        help="label column in reference TSVs",
    )
    l2_group.add_argument(
        "--level2-id-column", default="sequence_id",
        help="sequence ID column in reference TSVs",
    )
    l2_group.add_argument(
        "-k", "--level2-k", type=int, default=3,
        help="neighbors to retrieve per major class",
    )

    # -- Hardware / performance -----------------------------------------------
    hw_group = parser.add_argument_group("hardware/performance")
    hw_group.add_argument(
        "-d", "--device",
        help="torch device, e.g. cuda, cuda:0, cpu (auto-detect if omitted)",
    )
    hw_group.add_argument(
        "--mixed-precision", choices=["bf16", "fp16", "fp32"], default="fp32",
        help="mixed-precision mode",
    )
    hw_group.add_argument(
        "-b", "--batch-size", type=int, default=8,
        help="batch size for level 0/1 models",
    )
    hw_group.add_argument(
        "--max-length", type=int, default=1024,
        help="maximum sequence length (tokens)",
    )
    hw_group.add_argument(
        "--num-workers", type=int, default=0,
        help="dataloader worker processes",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ---- resolve input (positional vs --input flag) -------------------------
    fasta_path = args.input or args.input_flag
    if fasta_path is None:
        parser.error("a FASTA input file is required (positional or via -i/--input)")
    if not os.path.exists(fasta_path):
        parser.error(f"input file not found: {fasta_path}")

    # ---- basic validation ---------------------------------------------------
    if args.batch_size < 1:
        parser.error(f"--batch-size must be >= 1, got {args.batch_size}")
    if not 0 <= args.level0_threshold <= 1:
        parser.error(f"--level0-threshold must be in [0, 1], got {args.level0_threshold}")
    if not 0 <= args.level1_threshold <= 1:
        parser.error(f"--level1-threshold must be in [0, 1], got {args.level1_threshold}")

    # ---- default output name from input stem --------------------------------
    if args.output_name is None:
        args.output_name = os.path.splitext(os.path.basename(fasta_path))[0]

    log_gpu_count()

    pipeline = PredictionPipeline(
        device=args.device,
        mixed_precision=args.mixed_precision,
    )
    pipeline.predict(
        test_fasta=fasta_path,
        level0_model_path=args.level0_model,
        level1_model_path=args.level1_model,
        level0_threshold=args.level0_threshold,
        level1_thresholds=args.level1_thresholds,
        level1_thresholds_file=args.level1_thresholds_file,
        level1_global_threshold=args.level1_threshold,
        level2_model_path=args.level2_model,
        level2_families=args.level2_families,
        level2_faiss_dir=args.level2_faiss_dir,
        level2_label_tsv_dir=args.level2_label_tsv_dir,
        level2_label_column=args.level2_label_column,
        level2_id_column=args.level2_id_column,
        level2_k=args.level2_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        output_name=args.output_name,
        save_embeddings=args.save_embeddings,
        dataloader_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
