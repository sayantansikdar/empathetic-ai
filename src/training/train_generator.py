"""
Empathetic response generator trainer using LoRA-adapted GPT-2 XL.

Fine-tunes GPT-2 XL to generate empathetic responses conditioned on 
detected emotions from GoEmotions data.

Usage:
    python -m src.training.train_generator --lora --epochs 3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.response_generator import ResponseGenerator, EMPATHY_SYSTEM_CONTEXT, EMPATHY_PROMPT_TEMPLATE
from src.preprocessing.data_loader import load_goemotions, preprocess_dataset
from src.preprocessing.text_cleaner import SENTIMENT_LABELS, EMOTION_STARTERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Template Empathetic Responses (for training data synthesis)
# ──────────────────────────────────────────────────────────────────────────────

EMPATHETIC_TEMPLATES = {
    "positive": [
        "That's wonderful! Your happiness is contagious, and I'm thrilled to hear you're experiencing this.",
        "I can feel your excitement! It's great to see you in such a positive state.",
        "What a beautiful feeling! Cherish these moments of joy — you truly deserve them.",
        "That sounds amazing! I'm so happy for you. Keep riding this wave of positivity.",
        "Your enthusiasm is infectious! Thank you for sharing this wonderful feeling with me.",
    ],
    "negative": [
        "I'm truly sorry you're going through this. Your feelings are completely valid, and I'm here for you.",
        "That sounds incredibly difficult. Please know that it's okay to feel this way, and you don't have to face it alone.",
        "I can sense how much pain you're in. Take your time — healing isn't linear, and every step forward matters.",
        "I understand how overwhelming this must be. Remember that reaching out like this shows incredible strength.",
        "I hear you, and I want you to know that your feelings matter. Let's work through this together.",
    ],
    "neutral": [
        "Thank you for sharing. I appreciate you opening up about what's on your mind.",
        "I hear what you're saying. It sounds like you're processing a lot right now.",
        "That's an interesting perspective. Would you like to explore those thoughts further?",
        "I appreciate you sharing that with me. Sometimes just talking things through can help bring clarity.",
        "I understand. It sounds like you're reflecting on this thoughtfully.",
    ],
}


def build_training_examples(dataset, tokenizer, max_length: int = 256):
    """
    Build prompt-response training data from GoEmotions examples.
    
    For each GoEmotions example, creates a full training sequence:
        [system context] + [prompt with user text + emotion] + [empathetic response]
    """
    import random
    random.seed(42)
    
    texts = []
    
    for example in dataset:
        text = example["text"]
        label_id = example["label"]
        emotion = SENTIMENT_LABELS[label_id]
        
        # Select a random empathetic response template
        templates = EMPATHETIC_TEMPLATES.get(emotion, EMPATHETIC_TEMPLATES["neutral"])
        response = random.choice(templates)
        
        # Build full training sequence
        prompt = EMPATHY_PROMPT_TEMPLATE.format(
            user_text=text,
            emotion=emotion,
        )
        
        full_text = EMPATHY_SYSTEM_CONTEXT + prompt + " " + response
        texts.append(full_text)
    
    # Tokenize all at once
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    
    return Dataset.from_dict(tokenized)


def train_generator(args):
    """Main generator training function."""
    
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Training empathetic response generator")
    
    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_enabled = False
    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("empachat-response-generator")
        mlflow_enabled = True
        mlflow.start_run(run_name=f"generator-lora-e{args.epochs}")
        mlflow.log_params({
            "model_name": args.model_name,
            "use_lora": args.lora,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        })
    except ImportError:
        logger.warning("MLflow not available")
    
    # ── Initialize Generator ─────────────────────────────────────────────────
    generator = ResponseGenerator(
        model_name=args.model_name,
        use_lora=args.lora,
        use_quantization=args.quantize,
    )
    generator.load_model()
    model, tokenizer = generator.get_model_for_training()
    
    # ── Build Training Data ──────────────────────────────────────────────────
    logger.info("Building training data from GoEmotions...")
    
    raw_dataset = load_goemotions(cache_dir=args.data_dir)
    processed = preprocess_dataset(raw_dataset, num_classes=3)
    
    train_dataset = build_training_examples(
        processed["train"], tokenizer, max_length=args.max_length
    )
    val_dataset = build_training_examples(
        processed["validation"], tokenizer, max_length=args.max_length
    )
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # ── Training Arguments ───────────────────────────────────────────────────
    num_training_steps = (len(train_dataset) // args.batch_size) * args.epochs
    warmup_steps = int(0.1 * num_training_steps)

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
        save_total_limit=2,
        logging_steps=100,
        report_to=["mlflow"] if mlflow_enabled else ["none"],
        fp16=torch.cuda.is_available(),
        seed=42,
        remove_unused_columns=False,
    )
    
    # ── Data Collator ────────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Starting generator training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    train_time = time.time() - start_time
    logger.info(f"Generator training completed in {train_time:.1f}s ({train_time/60:.1f}min)")
    
    # ── Evaluate ─────────────────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    logger.info(f"Eval loss: {eval_results.get('eval_loss', 'N/A')}")
    
    # ── Log & Save ───────────────────────────────────────────────────────────
    if mlflow_enabled:
        mlflow.log_metrics({
            "eval_loss": eval_results.get("eval_loss", 0),
            "training_time_seconds": train_time,
        })
        mlflow.end_run()
    
    adapter_dir = f"{output_dir}/generator_lora"
    generator.model = model
    generator.save_adapter(adapter_dir)
    
    print(f"\n{'='*60}")
    print(f"  GENERATOR TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Eval Loss:     {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"  Training time: {train_time/60:.1f} minutes")
    print(f"  Adapter saved: {adapter_dir}")
    print(f"{'='*60}\n")
    
    return eval_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train empathetic response generator with LoRA"
    )
    parser.add_argument("--model_name", type=str, default="gpt2-xl")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./outputs/generator")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--mlflow_uri", type=str, default="./outputs/mlruns")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_generator(args)
