"""
Emotion classification trainer using LoRA-adapted GPT-2 XL.

Trains on GoEmotions dataset with HuggingFace Trainer + PEFT integration.
Target: 89% accuracy on 3-class simplified test set.

Usage:
    python -m src.training.train_classifier --lora --epochs 3 --num_classes 3
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.emotion_classifier import EmotionClassifier
from src.preprocessing.data_loader import create_tokenized_datasets
from src.preprocessing.text_cleaner import get_label_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """Compute accuracy, macro F1, and per-class F1 for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def train_classifier(args):
    """Main training function."""
    
    # ── Setup ────────────────────────────────────────────────────────────────
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training emotion classifier with {args.num_classes} classes")
    logger.info(f"LoRA: {args.lora}, Quantization: {args.quantize}")
    logger.info(f"Output: {output_dir}")
    
    # ── MLflow Tracking ──────────────────────────────────────────────────────
    mlflow_enabled = False
    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("empachat-emotion-classifier")
        mlflow_enabled = True
        mlflow.start_run(run_name=f"classifier-lora-{args.num_classes}cls-e{args.epochs}")
        mlflow.log_params({
            "model_name": args.model_name,
            "num_classes": args.num_classes,
            "use_lora": args.lora,
            "use_quantization": args.quantize,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_length": args.max_length,
        })
        logger.info("MLflow tracking enabled")
    except ImportError:
        logger.warning("MLflow not installed, skipping experiment tracking")
    
    # ── Initialize Classifier ────────────────────────────────────────────────
    # Determine device
    device = getattr(args, 'device', None)
    
    classifier = EmotionClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        use_lora=args.lora,
        use_quantization=args.quantize,
        device=device,
    )
    classifier.load_model()
    model, tokenizer = classifier.get_model_for_training()
    
    # Log parameter summary to MLflow
    if mlflow_enabled:
        total, trainable = 0, 0
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        mlflow.log_metrics({
            "total_params": total,
            "trainable_params": trainable,
            "param_reduction_pct": (1 - trainable / total) * 100,
        })
    
    # ── Load & Tokenize Dataset ──────────────────────────────────────────────
    logger.info("Loading and tokenizing GoEmotions dataset...")
    tokenized_datasets = create_tokenized_datasets(
        tokenizer=tokenizer,
        num_classes=args.num_classes,
        max_length=args.max_length,
        cache_dir=args.data_dir,
    )
    # Rename 'label' to 'labels' for HuggingFace Trainer compatibility
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # ── Training Arguments ───────────────────────────────────────────────────
    # Calculate warmup steps (10% of total)
    num_training_steps = (len(tokenized_datasets['train']) // args.batch_size) * args.epochs
    warmup_steps = int(0.1 * num_training_steps)

    import os
    os.environ["TENSORBOARD_LOGGING_DIR"] = f"{output_dir}/logs"

    # Determine if we should force CPU for Trainer
    force_cpu = (device == "cpu") if device else False

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=False,
        logging_steps=100,
        report_to=["mlflow"] if mlflow_enabled else ["none"],
        fp16=torch.cuda.is_available() and not force_cpu,
        dataloader_num_workers=0 if force_cpu else 2,
        use_cpu=force_cpu,
        seed=42,
    )
    
    # ── Data Collator ────────────────────────────────────────────────────────
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=args.max_length,
    )
    
    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s ({train_time/60:.1f}min)")
    
    # ── Evaluate on Test Set ─────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    
    accuracy = test_results.get("eval_accuracy", 0)
    f1_macro = test_results.get("eval_f1_macro", 0)
    f1_weighted = test_results.get("eval_f1_weighted", 0)
    
    logger.info(f"Test Accuracy:   {accuracy:.4f} ({accuracy*100:.1f}%)")
    logger.info(f"Test F1 (macro): {f1_macro:.4f}")
    logger.info(f"Test F1 (weighted): {f1_weighted:.4f}")
    
    # ── Detailed Classification Report ───────────────────────────────────────
    logger.info("Generating detailed classification report...")
    test_predictions = trainer.predict(tokenized_datasets["test"])
    preds = np.argmax(test_predictions.predictions, axis=-1)
    labels = test_predictions.label_ids
    
    label_names = get_label_names(args.num_classes)
    report = classification_report(labels, preds, target_names=label_names, digits=4)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save report
    report_path = Path(output_dir) / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Weighted: {f1_weighted:.4f}\n")
        f.write(f"Training Time: {train_time:.1f}s\n")
    
    # ── Log Final Metrics ────────────────────────────────────────────────────
    if mlflow_enabled:
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_weighted,
            "training_time_seconds": train_time,
        })
        mlflow.log_artifact(str(report_path))
        mlflow.end_run()
    
    # ── Save LoRA Adapter ────────────────────────────────────────────────────
    adapter_dir = f"{output_dir}/classifier_lora"
    classifier.model = model  # Update with trained model
    classifier.save_adapter(adapter_dir)
    logger.info(f"LoRA adapter saved to {adapter_dir}")
    
    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Accuracy:       {accuracy*100:.1f}%  (target: 89%)")
    print(f"  F1 (macro):     {f1_macro:.4f}")
    print(f"  F1 (weighted):  {f1_weighted:.4f}")
    print(f"  Training time:  {train_time/60:.1f} minutes")
    print(f"  Adapter saved:  {adapter_dir}")
    
    if accuracy >= 0.89:
        print(f"  ✅ TARGET ACHIEVED — {accuracy*100:.1f}% >= 89%")
    else:
        print(f"  ⚠️  Below target — {accuracy*100:.1f}% < 89%")
    print(f"{'='*60}\n")
    
    return test_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train emotion classifier with LoRA fine-tuning"
    )
    
    # Model
    parser.add_argument("--model_name", type=str, default="gpt2-xl",
                        help="HuggingFace model identifier")
    parser.add_argument("--num_classes", type=int, default=3, choices=[3, 27, 28],
                        help="Number of emotion classes (3=sentiment, 27/28=fine-grained)")
    
    # LoRA
    parser.add_argument("--lora", action="store_true", default=False,
                        help="Enable LoRA fine-tuning")
    parser.add_argument("--quantize", action="store_true", default=False,
                        help="Enable 4-bit QLoRA quantization")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device training batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max tokenization length")
    
    # Paths
    parser.add_argument("--output_dir", type=str, default="./outputs/classifier",
                        help="Output directory for checkpoints and adapter")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Dataset cache directory")
    parser.add_argument("--mlflow_uri", type=str, default="./outputs/mlruns",
                        help="MLflow tracking URI")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classifier(args)
