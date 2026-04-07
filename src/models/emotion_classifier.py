"""
Emotion classifier using GPT-2 XL with LoRA adapters.

Classifies input text → emotion label (3 or 27 classes).
Target: 89% accuracy on GoEmotions simplified 3-class test set.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2ForSequenceClassification,
)
from peft import PeftModel, get_peft_model

from .lora_config import (
    get_lora_config,
    get_quantization_config,
    print_param_summary,
)
from ..preprocessing.text_cleaner import (
    SENTIMENT_LABELS,
    EMOTION_LABELS,
    get_label_names,
)

logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL_NAME = "gpt2-xl"  # 1.5B params
FALLBACK_MODEL_NAME = "gpt2-medium"  # 355M params (for low-resource)


class EmotionClassifier:
    """
    GPT-2 based emotion classifier with optional LoRA adaptation.
    
    Usage:
        classifier = EmotionClassifier(num_classes=3, use_lora=True)
        classifier.load_model()
        label, scores = classifier.predict("I'm so happy today!")
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_classes: int = 3,
        use_lora: bool = True,
        use_quantization: bool = False,
        lora_adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            num_classes: Number of classification labels (3 or 28).
            use_lora: Apply LoRA adapters.
            use_quantization: Use 4-bit QLoRA quantization.
            lora_adapter_path: Path to pre-trained LoRA adapter weights.
            device: Target device (auto-detected if None).
            max_length: Max tokenization length.
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_lora = use_lora
        self.use_quantization = use_quantization
        self.lora_adapter_path = lora_adapter_path
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self.label_names = get_label_names(num_classes)
        
        logger.info(
            f"EmotionClassifier initialized — "
            f"model={model_name}, classes={num_classes}, "
            f"lora={use_lora}, quantize={use_quantization}, "
            f"device={self.device}"
        )
    
    def load_model(self) -> None:
        """Load the base model, tokenizer, and optionally apply LoRA."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        # Model loading kwargs
        model_kwargs = {
            "num_labels": self.num_classes,
            "problem_type": "single_label_classification",
        }
        
        # Quantization config
        if self.use_quantization and self.device == "cuda":
            model_kwargs["quantization_config"] = get_quantization_config()
            model_kwargs["device_map"] = "auto"
        
        # Load base classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        # Set pad token ID in model config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Apply LoRA
        if self.use_lora:
            if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
                # Load pre-trained adapter
                logger.info(f"Loading LoRA adapter from {self.lora_adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model, self.lora_adapter_path
                )
            else:
                # Apply fresh LoRA config
                lora_config = get_lora_config(task_type="SEQ_CLS")
                self.model = get_peft_model(self.model, lora_config)
            
            print_param_summary(self.model, "Emotion Classifier (LoRA)")
        
        # Move to device (if not using quantization with device_map)
        if not (self.use_quantization and self.device == "cuda"):
            self.model = self.model.to(self.device)
        
        logger.info("Model loaded successfully")
    
    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """
        Predict emotion for input text.
        
        Args:
            text: Input text to classify.
        
        Returns:
            Tuple of (predicted_label, confidence_scores_dict).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Handle edge cases
        if not text or not text.strip():
            return "neutral", {name: 0.0 for name in self.label_names}
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze()
        
        # Get prediction
        pred_idx = probs.argmax().item()
        pred_label = self.label_names[pred_idx]
        
        # Build confidence scores
        scores = {
            name: round(probs[i].item(), 4)
            for i, name in enumerate(self.label_names)
        }
        
        return pred_label, scores
    
    def predict_batch(self, texts: list[str]) -> list[tuple[str, dict[str, float]]]:
        """
        Predict emotions for a batch of texts.
        
        Args:
            texts: List of input texts.
        
        Returns:
            List of (predicted_label, confidence_scores) tuples.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
        
        # Process each prediction
        results = []
        for i in range(len(texts)):
            pred_idx = probs[i].argmax().item()
            pred_label = self.label_names[pred_idx]
            scores = {
                name: round(probs[i][j].item(), 4)
                for j, name in enumerate(self.label_names)
            }
            results.append((pred_label, scores))
        
        return results
    
    def save_adapter(self, output_dir: str) -> None:
        """Save the LoRA adapter weights."""
        if not self.use_lora:
            logger.warning("No LoRA adapter to save (use_lora=False)")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"LoRA adapter saved to {output_dir}")
    
    def get_model_for_training(self):
        """Return model and tokenizer for HuggingFace Trainer."""
        return self.model, self.tokenizer
