"""
Empathetic response generator using GPT-2 XL with LoRA adapters.

Takes classified emotion + user text → generates an empathetic reply.
The adapter is trained on real listener replies from EmpatheticDialogues
(see src/preprocessing/empathetic_dialogues.py), with the same compact prompt
used for training and inference.
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
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

from .lora_config import (
    get_lora_config,
    get_quantization_config,
    print_param_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gpt2-xl"

# ──────────────────────────────────────────────────────────────────────────────
# Prompt — identical at training and inference time
# ──────────────────────────────────────────────────────────────────────────────

EMPATHY_PROMPT_TEMPLATE = "Speaker ({emotion}): {user_text}\nListener:"


def build_prompt(user_text: str, emotion: str) -> str:
    """Format the conditioning prompt for one user message."""
    return EMPATHY_PROMPT_TEMPLATE.format(user_text=user_text.strip(), emotion=emotion)


class ResponseGenerator:
    """
    GPT-2 based empathetic response generator with LoRA.

    Usage:
        generator = ResponseGenerator(use_lora=True, lora_adapter_path="...")
        generator.load_model()
        reply = generator.generate_response("I lost my job today.", "negative")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        use_lora: bool = True,
        use_quantization: bool = False,
        lora_adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 128,
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

        self.generation_config = GenerationConfig(
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=None,  # set after the tokenizer loads
            eos_token_id=None,
        )

        logger.info(
            f"ResponseGenerator initialized — "
            f"model={model_name}, lora={use_lora}, device={self.device}"
        )

    def load_model(self) -> None:
        """Load the causal LM, tokenizer and (new or saved) LoRA adapter."""
        logger.info(f"Loading generator model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Right padding for training: GPT-2 uses absolute positions, so left
        # padding would shift every position relative to unpadded inference.
        self.tokenizer.padding_side = "right"

        model_kwargs = {}
        quantized = self.use_quantization and self.device == "cuda"
        if quantized:
            model_kwargs["quantization_config"] = get_quantization_config()
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Stop at end-of-text or at the end of the listener's line.
        newline_id = self.tokenizer.encode("\n")[0]
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = [self.tokenizer.eos_token_id, newline_id]

        if self.use_lora:
            if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
                logger.info(f"Loading LoRA adapter from {self.lora_adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
            else:
                if quantized:
                    self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
                self.model = get_peft_model(self.model, get_lora_config(task_type="CAUSAL_LM"))
            print_param_summary(self.model, "Response Generator (LoRA)")

        if not quantized:
            self.model = self.model.to(self.device)

        logger.info("Generator model loaded successfully")

    def build_prompt(self, user_text: str, emotion: str) -> str:
        return build_prompt(user_text, emotion)

    def generate_response(
        self,
        user_text: str,
        emotion: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate an empathetic reply to `user_text` given its emotion."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not user_text or not user_text.strip():
            return ""

        self.model.eval()
        inputs = self.tokenizer(
            self.build_prompt(user_text, emotion),
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        gen_config = GenerationConfig(**{
            **self.generation_config.to_dict(),
            "max_new_tokens": max_new_tokens or self.generation_config.max_new_tokens,
            "temperature": temperature or self.generation_config.temperature,
            "top_p": top_p or self.generation_config.top_p,
        })

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)

        reply = self.tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
        return self._postprocess_response(reply)

    @staticmethod
    def _postprocess_response(response: str) -> str:
        """Keep the first line and drop a trailing half-sentence, if any."""
        response = response.split("\n")[0].strip()
        if response and response[-1] not in ".!?\"')":
            cut = max(response.rfind(p) for p in (". ", "! ", "? "))
            if cut > 0:
                response = response[: cut + 1]
        return response

    def generate_batch(self, user_texts: list[str], emotions: list[str]) -> list[str]:
        return [self.generate_response(t, e) for t, e in zip(user_texts, emotions)]

    def save_adapter(self, output_dir: str) -> None:
        """Save the LoRA adapter weights and tokenizer."""
        if not self.use_lora:
            logger.warning("No LoRA adapter to save")
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Generator LoRA adapter saved to {output_dir}")

    def get_model_for_training(self):
        """Return model and tokenizer for the HuggingFace Trainer."""
        return self.model, self.tokenizer
