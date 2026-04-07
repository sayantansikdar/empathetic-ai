"""
Empathetic response generator using GPT-2 XL with LoRA adapters.

Takes classified emotion + user text → generates contextually empathetic reply.
Uses controlled generation (temperature, top_p, repetition_penalty).
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel, get_peft_model

from .lora_config import (
    get_lora_config,
    get_quantization_config,
    print_param_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gpt2-xl"

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────────────────────────────────────

EMPATHY_PROMPT_TEMPLATE = (
    "The user said: \"{user_text}\"\n"
    "Their detected emotion is: {emotion}.\n"
    "Generate a warm, empathetic, and supportive response:\n"
    "Response:"
)

EMPATHY_SYSTEM_CONTEXT = (
    "You are EmpaChat, a compassionate and empathetic AI assistant. "
    "You listen carefully to the user's feelings and respond with "
    "understanding, validation, and gentle support. "
    "Keep responses concise (2-3 sentences) and genuine.\n\n"
)

# Emotion-specific response starters for enhanced empathy
EMOTION_STARTERS = {
    "positive": [
        "That's wonderful to hear!",
        "I'm so glad you're feeling this way!",
        "That sounds amazing!",
    ],
    "negative": [
        "I understand this is difficult.",
        "I'm sorry you're going through this.",
        "That sounds really tough, and your feelings are valid.",
    ],
    "neutral": [
        "I appreciate you sharing that.",
        "Thank you for telling me about this.",
        "I hear you.",
    ],
}


class ResponseGenerator:
    """
    GPT-2 based empathetic response generator with LoRA.
    
    Usage:
        generator = ResponseGenerator(use_lora=True)
        generator.load_model()
        response = generator.generate_response(
            "I just lost my job", emotion="negative"
        )
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        use_lora: bool = True,
        use_quantization: bool = False,
        lora_adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.use_lora = use_lora
        self.use_quantization = use_quantization
        self.lora_adapter_path = lora_adapter_path
        self.max_length = max_length
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        
        # Default generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=None,  # Set after tokenizer loads
            eos_token_id=None,
        )
        
        logger.info(
            f"ResponseGenerator initialized — "
            f"model={model_name}, lora={use_lora}, device={self.device}"
        )
    
    def load_model(self) -> None:
        """Load GPT-2 XL for causal LM with optional LoRA."""
        logger.info(f"Loading generator model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        # Model loading kwargs
        model_kwargs = {}
        
        if self.use_quantization and self.device == "cuda":
            model_kwargs["quantization_config"] = get_quantization_config()
            model_kwargs["device_map"] = "auto"
        
        # Load base causal LM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        # Update generation config with tokenizer IDs
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Apply LoRA
        if self.use_lora:
            if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
                logger.info(f"Loading LoRA adapter from {self.lora_adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model, self.lora_adapter_path
                )
            else:
                lora_config = get_lora_config(task_type="CAUSAL_LM")
                self.model = get_peft_model(self.model, lora_config)
            
            print_param_summary(self.model, "Response Generator (LoRA)")
        
        # Move to device
        if not (self.use_quantization and self.device == "cuda"):
            self.model = self.model.to(self.device)
        
        logger.info("Generator model loaded successfully")
    
    def build_prompt(self, user_text: str, emotion: str) -> str:
        """
        Build the full prompt for empathetic response generation.
        
        Args:
            user_text: What the user said.
            emotion: Detected emotion label.
        
        Returns:
            Formatted prompt string.
        """
        prompt = EMPATHY_SYSTEM_CONTEXT + EMPATHY_PROMPT_TEMPLATE.format(
            user_text=user_text,
            emotion=emotion,
        )
        return prompt
    
    def generate_response(
        self,
        user_text: str,
        emotion: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate an empathetic response.
        
        Args:
            user_text: User's message.
            emotion: Detected emotion (e.g., "positive", "negative", "neutral").
            temperature: Override generation temperature.
            top_p: Override nucleus sampling threshold.
            max_new_tokens: Override max tokens to generate.
        
        Returns:
            Generated empathetic response text.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Handle edge cases
        if not user_text or not user_text.strip():
            return "I'm here to listen. Could you share what's on your mind?"
        
        # Build prompt
        prompt = self.build_prompt(user_text, emotion)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)
        
        prompt_length = inputs["input_ids"].shape[1]
        
        # Override generation params if provided
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.generation_config.max_new_tokens,
            temperature=temperature or self.generation_config.temperature,
            top_p=top_p or self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            repetition_penalty=self.generation_config.repetition_penalty,
            do_sample=True,
            pad_token_id=self.generation_config.pad_token_id,
            eos_token_id=self.generation_config.eos_token_id,
        )
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )
        
        # Decode only the generated tokens (exclude prompt)
        generated_ids = output_ids[0, prompt_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Post-process: clean up the response
        response = self._postprocess_response(response, emotion)
        
        return response
    
    def _postprocess_response(self, response: str, emotion: str) -> str:
        """Clean and ensure quality of generated response."""
        # Remove any trailing incomplete sentences
        response = response.strip()
        
        # If response ends mid-sentence, truncate at last period/question/exclamation
        if response and response[-1] not in ".!?\"'":
            for punct in [". ", "! ", "? "]:
                last_idx = response.rfind(punct)
                if last_idx > 0:
                    response = response[:last_idx + 1]
                    break
        
        # If the response is empty or too short, use a fallback
        if len(response.split()) < 3:
            import random
            starters = EMOTION_STARTERS.get(emotion, EMOTION_STARTERS["neutral"])
            response = random.choice(starters) + " I'm here for you and want to help in any way I can."
        
        # Truncate if too long (keep it concise)
        sentences = response.replace("! ", "!|").replace(". ", ".|").replace("? ", "?|").split("|")
        if len(sentences) > 4:
            response = " ".join(sentences[:3]).strip()
        
        return response
    
    def generate_batch(
        self,
        user_texts: list[str],
        emotions: list[str],
    ) -> list[str]:
        """Generate empathetic responses for a batch of inputs."""
        return [
            self.generate_response(text, emotion)
            for text, emotion in zip(user_texts, emotions)
        ]
    
    def save_adapter(self, output_dir: str) -> None:
        """Save the LoRA adapter weights."""
        if not self.use_lora:
            logger.warning("No LoRA adapter to save")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Generator LoRA adapter saved to {output_dir}")
    
    def get_model_for_training(self):
        """Return model and tokenizer for HuggingFace Trainer."""
        return self.model, self.tokenizer


def create_training_data_from_goemotions(
    dataset,
    tokenizer,
    max_length: int = 256,
) -> list[dict]:
    """
    Convert GoEmotions examples into prompt-response pairs for generator training.
    
    Uses emotion-conditioned prompts with template responses.
    
    Args:
        dataset: Preprocessed GoEmotions dataset split.
        tokenizer: GPT-2 tokenizer.
        max_length: Max sequence length.
    
    Returns:
        List of tokenized training examples.
    """
    from ..preprocessing.text_cleaner import SENTIMENT_LABELS
    
    training_examples = []
    
    for example in dataset:
        text = example["text"]
        label_id = example["label"]
        emotion = SENTIMENT_LABELS[label_id]
        
        # Build prompt + expected response format
        prompt = EMPATHY_PROMPT_TEMPLATE.format(
            user_text=text,
            emotion=emotion,
        )
        
        # For training, we concatenate prompt + response template
        full_text = EMPATHY_SYSTEM_CONTEXT + prompt
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        training_examples.append(tokenized)
    
    return training_examples
