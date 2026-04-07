"""
GoEmotions dataset loading, preprocessing, and tokenization pipeline.
"""

import logging
from typing import Optional

from datasets import load_dataset, DatasetDict, Dataset
from transformers import PreTrainedTokenizer

from .text_cleaner import (
    clean_text,
    map_emotion_id_to_sentiment_id,
    EMOTION_LABELS,
    SENTIMENT_LABELS,
)

logger = logging.getLogger(__name__)


def load_goemotions(cache_dir: str = "./data") -> DatasetDict:
    """
    Load GoEmotions dataset from HuggingFace Hub.
    
    The dataset has train/validation/test splits with columns:
        - text: str (Reddit comment)
        - labels: list[int] (multi-label emotion IDs, 0-27)
    
    Args:
        cache_dir: Directory to cache the downloaded dataset.
    
    Returns:
        DatasetDict with train/validation/test splits.
    """
    logger.info("Loading GoEmotions dataset from HuggingFace Hub...")
    
    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified",
        cache_dir=cache_dir,
    )
    
    logger.info(
        f"Dataset loaded — "
        f"Train: {len(dataset['train'])}, "
        f"Validation: {len(dataset['validation'])}, "
        f"Test: {len(dataset['test'])}"
    )
    
    return dataset


def _preprocess_example(example: dict, num_classes: int = 3) -> dict:
    """
    Preprocess a single GoEmotions example.
    
    - Cleans text
    - Converts multi-label to single-label (takes first label)
    - Maps to sentiment if num_classes == 3
    """
    # Clean the raw text
    cleaned = clean_text(example["text"])
    
    # GoEmotions labels is a list (multi-label); take the first label
    raw_labels = example["labels"]
    primary_label = raw_labels[0] if raw_labels else 27  # Default to 'neutral'
    
    # Map to sentiment if using simplified 3-class
    if num_classes == 3:
        label = map_emotion_id_to_sentiment_id(primary_label)
    else:
        label = primary_label
    
    return {
        "text": cleaned,
        "label": label,
        "original_text": example["text"],
    }


def preprocess_dataset(
    dataset: DatasetDict,
    num_classes: int = 3,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Apply preprocessing to the entire dataset.
    
    Args:
        dataset: Raw GoEmotions DatasetDict.
        num_classes: 3 (sentiment) or 28 (fine-grained).
        num_proc: Number of processes for parallel preprocessing.
    
    Returns:
        Preprocessed DatasetDict with 'text', 'label' columns.
    """
    logger.info(f"Preprocessing dataset with {num_classes}-class taxonomy...")
    
    processed = dataset.map(
        lambda ex: _preprocess_example(ex, num_classes=num_classes),
        num_proc=num_proc,
        desc="Preprocessing",
    )
    
    # Remove original columns that are no longer needed
    cols_to_remove = [c for c in processed["train"].column_names 
                      if c not in ("text", "label", "original_text")]
    processed = processed.remove_columns(cols_to_remove)
    
    # Filter out empty texts
    processed = processed.filter(
        lambda ex: len(ex["text"].strip()) > 0,
        num_proc=num_proc,
        desc="Filtering empty texts",
    )
    
    # Log class distribution
    for split in processed:
        labels = processed[split]["label"]
        label_names = SENTIMENT_LABELS if num_classes == 3 else EMOTION_LABELS
        dist = {}
        for lbl in labels:
            name = label_names[lbl] if lbl < len(label_names) else f"unknown_{lbl}"
            dist[name] = dist.get(name, 0) + 1
        logger.info(f"{split} split distribution: {dist}")
    
    return processed


def create_tokenized_datasets(
    tokenizer: PreTrainedTokenizer,
    num_classes: int = 3,
    max_length: int = 128,
    cache_dir: str = "./data",
    num_proc: int = 4,
) -> DatasetDict:
    """
    Full pipeline: load → preprocess → tokenize GoEmotions.
    
    Args:
        tokenizer: GPT-2 tokenizer (left-padded).
        num_classes: 3 or 28.
        max_length: Max sequence length for tokenization.
        cache_dir: Dataset cache directory.
        num_proc: Number of parallel workers.
    
    Returns:
        Tokenized DatasetDict ready for training.
    """
    # Ensure pad token is set (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Left padding for decoder-only models (classification uses last token)
    tokenizer.padding_side = "left"
    
    # Load and preprocess
    raw_dataset = load_goemotions(cache_dir=cache_dir)
    processed = preprocess_dataset(raw_dataset, num_classes=num_classes, num_proc=num_proc)
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return lists for Dataset.map()
        )
    
    tokenized = processed.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing",
        remove_columns=["text", "original_text"],
    )
    
    # Set format for PyTorch
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    logger.info(
        f"Tokenized dataset ready — "
        f"Train: {len(tokenized['train'])}, "
        f"Val: {len(tokenized['validation'])}, "
        f"Test: {len(tokenized['test'])}, "
        f"Max length: {max_length}"
    )
    
    return tokenized
