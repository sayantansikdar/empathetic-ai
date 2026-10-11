#!/usr/bin/env python3
"""
EmpaChat — Main Training Entry Point

One-liner: python train.py --lora --epochs 3

Orchestrates:
  1. Emotion classification training (GoEmotions + LoRA)
  2. Response generator training (empathetic prompts + LoRA)
  3. A/B testing simulation
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# The log file lives in outputs/, so the folder must exist before logging starts.
Path("outputs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/training.log", mode="a"),
    ],
)
logger = logging.getLogger("empachat")


def main():
    parser = argparse.ArgumentParser(
        description="EmpaChat — Empathetic Response AI Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --lora --epochs 3
  python train.py --lora --quantize --epochs 5 --batch_size 4
  python train.py --lora --epochs 3 --skip_generator --skip_ab_test
  python train.py --lora --epochs 3 --model gpt2-medium  # lighter model
        """,
    )
    
    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="gpt2-xl",
        help="Base model (gpt2, gpt2-medium, gpt2-large, gpt2-xl)"
    )
    parser.add_argument(
        "--num_classes", type=int, default=3, choices=[3, 27, 28],
        help="Number of emotion classes (3=sentiment, 27/28=fine-grained)"
    )
    
    # ── LoRA ─────────────────────────────────────────────────────────────────
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit QLoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Classifier max sequence length (GoEmotions comments are short)")
    parser.add_argument("--gen_max_length", type=int, default=128,
                        help="Generator max sequence length (prompt + reply)")
    parser.add_argument("--gen_batch_size", type=int, default=None,
                        help="Generator batch size (defaults to --batch_size)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training examples per stage (time-boxed runs)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Cap evaluation examples per stage (smoke tests)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint/eval interval; checkpoints allow resuming a cut-off run")
    parser.add_argument("--hub_repo", type=str, default=None,
                        help="Push each adapter to this Hugging Face repo as soon as its stage finishes")
    
    # ── Pipeline Control ─────────────────────────────────────────────────────
    parser.add_argument("--skip_classifier", action="store_true",
                        help="Skip classifier training")
    parser.add_argument("--skip_generator", action="store_true",
                        help="Skip generator training")
    parser.add_argument("--skip_ab_test", action="store_true",
                        help="Skip A/B testing")
    parser.add_argument("--ab_samples", type=int, default=100,
                        help="Number of A/B test samples")
    parser.add_argument("--use_gpt4_judge", action="store_true",
                        help="Use GPT-4 as A/B test judge")
    
    # ── Device ────────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu, cuda, mps (auto-detect if not set)")
    
    # ── Paths ────────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Dataset cache")
    parser.add_argument("--mlflow_uri", type=str, default="./outputs/mlruns",
                        help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    import torch
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            # Force CPU — MPS on Mac runs out of memory for large models
            args.device = "cpu"
    
    # Set env vars to force Trainer to use the correct device
    import os
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # Disable MPS for Trainer (it auto-detects MPS and OOMs on large models)
    if args.device != "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    
    # Ensure output dirs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    overall_start = time.time()
    
    print(r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              EmpaChat Training Pipeline                    ║
    ║          Empathetic Response AI with LoRA                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print(f"  Model:        {args.model}")
    print(f"  Device:       {args.device}")
    print(f"  Classes:      {args.num_classes}")
    print(f"  LoRA:         {'✅' if args.lora else '❌'}")
    print(f"  QLoRA 4-bit:  {'✅' if args.quantize else '❌'}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Output:       {args.output_dir}")
    print()
    
    # ═══════════════════════════════════════════════════════════════════════
    # Stage 1: Emotion Classifier
    # ═══════════════════════════════════════════════════════════════════════
    if not args.skip_classifier:
        print("\n" + "─"*60)
        print("  STAGE 1: Emotion Classifier Training")
        print("─"*60 + "\n")
        
        from src.training.train_classifier import train_classifier
        
        class ClassifierArgs:
            model_name = args.model
            num_classes = args.num_classes
            lora = args.lora
            quantize = args.quantize
            lora_r = args.lora_r
            lora_alpha = args.lora_alpha
            epochs = args.epochs
            batch_size = args.batch_size
            lr = args.lr
            max_length = args.max_length
            device = args.device
            output_dir = f"{args.output_dir}/classifier"
            data_dir = args.data_dir
            mlflow_uri = args.mlflow_uri
            max_train_samples = args.max_train_samples
            max_eval_samples = args.max_eval_samples
            save_steps = args.save_steps
            hub_repo = args.hub_repo
        
        classifier_results = train_classifier(ClassifierArgs())
        logger.info(f"Classifier results: {classifier_results}")
    else:
        print("\n  ⏭️  Skipping classifier training")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Stage 2: Response Generator
    # ═══════════════════════════════════════════════════════════════════════
    if not args.skip_generator:
        print("\n" + "─"*60)
        print("  STAGE 2: Response Generator Training")
        print("─"*60 + "\n")
        
        from src.training.train_generator import train_generator
        
        class GeneratorArgs:
            model_name = args.model
            lora = args.lora
            quantize = args.quantize
            epochs = args.epochs
            batch_size = args.gen_batch_size or args.batch_size
            lr = args.lr
            max_length = args.gen_max_length
            device = args.device
            output_dir = f"{args.output_dir}/generator"
            data_dir = args.data_dir
            mlflow_uri = args.mlflow_uri
            max_train_samples = args.max_train_samples
            max_eval_samples = args.max_eval_samples
            save_steps = args.save_steps
            hub_repo = args.hub_repo
        
        generator_results = train_generator(GeneratorArgs())
        logger.info(f"Generator results: {generator_results}")
    else:
        print("\n  ⏭️  Skipping generator training")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Stage 3: A/B Testing
    # ═══════════════════════════════════════════════════════════════════════
    if not args.skip_ab_test:
        print("\n" + "─"*60)
        print("  STAGE 3: A/B Testing Simulation")
        print("─"*60 + "\n")
        
        from src.training.ab_testing import run_ab_test
        
        ab_results = run_ab_test(
            num_samples=args.ab_samples,
            use_gpt4=args.use_gpt4_judge,
            output_dir=f"{args.output_dir}/ab_test",
        )
        logger.info(f"A/B test results: {ab_results}")
    else:
        print("\n  ⏭️  Skipping A/B testing")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - overall_start
    
    print(f"\n{'═'*60}")
    print(f"  EMPACHAT TRAINING PIPELINE COMPLETE")
    print(f"{'═'*60}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Output dir: {args.output_dir}")
    print(f"")
    print(f"  Next steps:")
    print(f"    1. Start API:  uvicorn src.api.main:app --reload")
    print(f"    2. Start Demo: python demo/gradio_app.py")
    print(f"    3. View MLflow: mlflow ui --backend-store-uri {args.mlflow_uri}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
