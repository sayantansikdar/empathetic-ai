"""
Evaluation metrics for emotion classification and response quality.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Predicted class indices.
        labels: True class indices.
        label_names: Human-readable class names.
    
    Returns:
        Dictionary of metrics.
    """
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
    }
    
    # Add per-class F1
    if label_names:
        for i, name in enumerate(label_names):
            if i < len(per_class_f1):
                metrics[f"f1_{name}"] = float(per_class_f1[i])
    
    return metrics


def generate_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a detailed classification report and optionally save to file.
    
    Returns:
        Report string.
    """
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        digits=4,
    )
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Classification report saved to {output_path}")
    
    return report


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[list[str]] = None,
) -> np.ndarray:
    """Compute confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    return cm


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[list[str]] = None,
    output_path: str = "./outputs/confusion_matrix.png",
    title: str = "Emotion Classification Confusion Matrix",
) -> str:
    """
    Plot and save a confusion matrix heatmap.
    
    Returns:
        Path to saved image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed, skipping plot")
        return ""
    
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names or range(cm.shape[0]),
        yticklabels=label_names or range(cm.shape[0]),
        ax=ax,
    )
    
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")
    return output_path


def compute_response_quality_metrics(
    responses: list[str],
    reference_responses: Optional[list[str]] = None,
) -> dict:
    """
    Compute quality metrics for generated responses.
    
    Metrics:
    - Average response length
    - Unique word ratio (vocab diversity)
    - Repetition rate
    """
    if not responses:
        return {}
    
    # Length stats
    lengths = [len(r.split()) for r in responses]
    avg_length = np.mean(lengths)
    
    # Vocabulary diversity
    all_words = " ".join(responses).lower().split()
    unique_ratio = len(set(all_words)) / max(len(all_words), 1)
    
    # Repetition rate (% of responses with repeated phrases)
    repetition_count = 0
    for r in responses:
        words = r.lower().split()
        if len(words) > 5:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            if len(bigrams) != len(set(bigrams)):
                repetition_count += 1
    
    repetition_rate = repetition_count / len(responses)
    
    return {
        "avg_response_length": float(avg_length),
        "vocab_diversity": float(unique_ratio),
        "repetition_rate": float(repetition_rate),
        "num_responses": len(responses),
    }
