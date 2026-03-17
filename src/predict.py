import argparse
from caalm import CAALMPredictor
from utils import log_gpu_count

def main():
    parser = argparse.ArgumentParser(
        description='CAALM: Predict CAZymes and CAZyme classes from protein sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--level0-model', dest='level0_model',
                        help='Path to level0 classification model')
    parser.add_argument('--level1-model', dest='level1_model',
                        help='Path to level1 classification model')
    parser.add_argument('--input', required=True, help='Path to input FASTA file')
    
    parser.add_argument('--level0-threshold', dest='level0_threshold',
                        type=float, default=0.5,
                        help='Threshold for level0 classification')
    
    parser.add_argument('--level1-threshold', dest='level1_threshold',
                        type=float, default=0.5,
                        help='Global threshold for level1 classification')
    parser.add_argument('--level1-thresholds', dest='level1_thresholds',
                        type=float, nargs='*',
                        help='Per-class thresholds (6 values)')
    parser.add_argument('--level1-thresholds-file', dest='level1_thresholds_file',
                        help='JSON file with per-class thresholds')
    
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for both models')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                        help='Device (auto-detect if not specified)')
    parser.add_argument('--mixed-precision', choices=['bf16', 'fp16', 'fp32'], default='fp32',
                        help='Mixed precision')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    parser.add_argument('--output-dir', default='./outputs',
                        help='Output directory')
    parser.add_argument('--output-name', default='test',
                        help='Prefix for output files')
    parser.add_argument('--save-embeddings', action='store_true', default=False,
                        help='Save embeddings')
    
    args = parser.parse_args()
    
    log_gpu_count()
    
    predictor = CAALMPredictor(
        device=args.device,
        mixed_precision=args.mixed_precision
    )
    
    predictor.predict(
        test_fasta=args.input,
        level0_model_path=args.level0_model,
        level1_model_path=args.level1_model,
        level0_threshold=args.level0_threshold,
        level1_thresholds=args.level1_thresholds,
        level1_thresholds_file=args.level1_thresholds_file,
        level1_global_threshold=args.level1_threshold,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        output_name=args.output_name,
        save_embeddings=args.save_embeddings,
        dataloader_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
