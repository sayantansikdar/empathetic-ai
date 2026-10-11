"""
Empathetic response generator trainer (GPT-2 XL + LoRA).

Trains on real listener replies from EmpatheticDialogues: each speaker
message, conditioned on its 3-class sentiment, is paired with the human
listener's reply, and the loss is computed on the reply only. Measured
validation loss / perplexity and sample generations are written next to the
adapter (metrics.json, samples.json) and optionally pushed to the Hub.

Usage:
    python -m src.training.train_generator --lora --epochs 1
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.response_generator import ResponseGenerator, build_prompt
from src.preprocessing.empathetic_dialogues import ED_DATASET, load_empathetic_pairs, tokenize_pairs
from src.utils.hub import push_adapter, write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Fixed prompts for qualitative samples saved alongside the metrics.
SAMPLE_PROMPTS = [
    ("I just got promoted at work and I'm so excited about it!", "positive"),
    ("My grandmother passed away last week and I miss her so much.", "negative"),
    ("I failed my exam even though I studied really hard.", "negative"),
    ("I've been thinking about my old school friends lately.", "neutral"),
    ("Someone left me the kindest note today, it made my whole week!", "positive"),
    ("I feel completely overwhelmed and don't know what to do anymore.", "negative"),
]


def train_generator(args):
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = getattr(args, "device", None)
    force_cpu = device == "cpu"

    # ── MLflow (optional) ────────────────────────────────────────────────────
    mlflow_enabled = False
    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("empachat-response-generator")
        mlflow.start_run(run_name=f"generator-lora-e{args.epochs}")
        mlflow.log_params({
            "model_name": args.model_name, "dataset": ED_DATASET, "use_lora": args.lora,
            "use_quantization": args.quantize, "learning_rate": args.lr,
            "batch_size": args.batch_size, "epochs": args.epochs, "max_length": args.max_length,
        })
        mlflow_enabled = True
    except ImportError:
        logger.warning("MLflow not installed, skipping experiment tracking")

    # ── Model ────────────────────────────────────────────────────────────────
    generator = ResponseGenerator(
        model_name=args.model_name,
        use_lora=args.lora,
        use_quantization=args.quantize,
        device=device,
        max_length=args.max_length,
    )
    generator.load_model()
    model, tokenizer = generator.get_model_for_training()

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info(f"Building training pairs from {ED_DATASET}...")
    pairs = load_empathetic_pairs(
        cache_dir=args.data_dir,
        max_train_samples=getattr(args, "max_train_samples", None),
        max_eval_samples=getattr(args, "max_eval_samples", None),
    )
    train_dataset = tokenize_pairs(pairs["train"], tokenizer, build_prompt, args.max_length)
    val_dataset = tokenize_pairs(pairs["validation"], tokenizer, build_prompt, args.max_length)
    logger.info(f"Training pairs: {len(train_dataset)} | validation pairs: {len(val_dataset)}")

    # ── Training ─────────────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_dataset) // args.batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=int(0.05 * steps_per_epoch * args.epochs),
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=getattr(args, "save_steps", 500),
        save_strategy="steps",
        save_steps=getattr(args, "save_steps", 500),
        save_total_limit=2,
        logging_steps=50,
        report_to=["mlflow"] if mlflow_enabled else ["none"],
        fp16=torch.cuda.is_available() and not force_cpu,
        use_cpu=force_cpu,
        dataloader_num_workers=0,
        seed=42,
    )
    # Pads inputs with EOS (right side) and labels with -100, so padding never
    # contributes to the loss.
    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    ckpts = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    last_ckpt = str(ckpts[-1]) if ckpts else None
    if last_ckpt:
        logger.info(f"Resuming from {last_ckpt}")

    start = time.time()
    trainer.train(resume_from_checkpoint=last_ckpt)
    train_time = time.time() - start

    eval_results = trainer.evaluate()
    eval_loss = float(eval_results.get("eval_loss", float("nan")))
    perplexity = math.exp(eval_loss) if eval_loss < 50 else float("inf")
    logger.info(f"Validation loss {eval_loss:.4f} | perplexity {perplexity:.2f}")

    # ── Qualitative samples ─────────────────────────────────────────────────
    generator.model = model
    torch.manual_seed(42)
    samples = [
        {"user_text": text, "emotion": emo, "reply": generator.generate_response(text, emo)}
        for text, emo in SAMPLE_PROMPTS
    ]
    for s in samples:
        logger.info(f"[{s['emotion']}] {s['user_text']}\n    → {s['reply']}")

    # ── Save, record, publish ───────────────────────────────────────────────
    adapter_dir = f"{output_dir}/generator_lora"
    generator.save_adapter(adapter_dir)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {
        "task": "Empathetic reply generation (EmpatheticDialogues, reply-only loss)",
        "dataset": ED_DATASET,
        "base_model": args.model_name,
        "quantized_4bit": bool(args.quantize),
        "val_loss": round(eval_loss, 4),
        "val_perplexity": round(perplexity, 2),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "train_minutes": round(train_time / 60, 1),
    }
    write_json(f"{adapter_dir}/metrics.json", metrics)
    write_json(f"{adapter_dir}/samples.json", {"samples": samples})
    if mlflow_enabled:
        mlflow.log_metrics({"val_loss": eval_loss, "val_perplexity": perplexity, "training_time_seconds": train_time})
        mlflow.end_run()
    push_adapter(adapter_dir, getattr(args, "hub_repo", None), "generator",
                 f"Generator adapter — val perplexity {perplexity:.2f}")

    print(f"\n{'='*60}")
    print("  GENERATOR TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Val loss:       {eval_loss:.4f}  (perplexity {perplexity:.2f})")
    print(f"  Training time:  {train_time/60:.1f} minutes")
    print(f"  Adapter saved:  {adapter_dir}")
    print(f"{'='*60}\n")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the empathetic response generator with LoRA")
    parser.add_argument("--model_name", type=str, default="gpt2-xl")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hub_repo", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/generator")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--mlflow_uri", type=str, default="./outputs/mlruns")
    return parser.parse_args()


if __name__ == "__main__":
    train_generator(parse_args())
