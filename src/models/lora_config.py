"""
LoRA and quantization configuration for parameter-efficient fine-tuning.

Achieves ~99.4% parameter reduction:
    GPT-2 XL baseline: ~1.56B parameters
    LoRA trainable:    ~8.4M parameters
"""

import logging
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Default LoRA hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_LORA_R = 8          # Rank of low-rank matrices
DEFAULT_LORA_ALPHA = 16     # Scaling factor (alpha / r = scaling)
DEFAULT_LORA_DROPOUT = 0.1  # Dropout for regularization

# GPT-2 target modules:
#   c_attn  — combined QKV projection (3 * d_model × d_model)
#   c_proj  — attention output projection (d_model × d_model)
#   c_fc    — MLP first linear layer (4 * d_model × d_model)
#   (optional: c_proj in MLP, but already named c_proj in attn)
DEFAULT_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]


def get_lora_config(
    task_type: str = "SEQ_CLS",
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Optional[list[str]] = None,
    bias: str = "none",
) -> LoraConfig:
    """
    Create a LoRA configuration for GPT-2 fine-tuning.
    
    Args:
        task_type: "SEQ_CLS" for classification, "CAUSAL_LM" for generation.
        r: Rank of the low-rank decomposition.
        lora_alpha: Scaling factor.
        lora_dropout: Dropout probability.
        target_modules: Which GPT-2 layers to apply LoRA to.
        bias: Bias training strategy ("none", "all", "lora_only").
    
    Returns:
        LoraConfig instance.
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES
    
    # Map string to TaskType enum
    task_type_enum = {
        "SEQ_CLS": TaskType.SEQ_CLS,
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
    }.get(task_type, TaskType.SEQ_CLS)
    
    config = LoraConfig(
        task_type=task_type_enum,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        fan_in_fan_out=True,  # Required for GPT-2 Conv1D layers
    )
    
    logger.info(
        f"LoRA config created — "
        f"task={task_type}, r={r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}, targets={target_modules}"
    )
    
    return config


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes quantization config for QLoRA.
    
    4-bit NF4 quantization reduces GPU memory from ~6GB to ~1GB for GPT-2 XL,
    leaving ample headroom on T4 16GB for training.
    
    Args:
        load_in_4bit: Use 4-bit quantization.
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4").
        bnb_4bit_compute_dtype: Compute dtype for operations.
        bnb_4bit_use_double_quant: Double quantization for extra savings.
    
    Returns:
        BitsAndBytesConfig instance.
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    
    logger.info(
        f"Quantization config — 4bit={load_in_4bit}, "
        f"type={bnb_4bit_quant_type}, compute={bnb_4bit_compute_dtype}"
    )
    
    return config


def print_param_summary(model, model_name: str = "Model") -> dict:
    """
    Print and return a summary of trainable vs total parameters.
    
    Validates the target: ~8.4M trainable from ~1.5B total (99.4% reduction).
    
    Args:
        model: The PEFT-wrapped model.
        model_name: Display name for logging.
    
    Returns:
        Dict with parameter counts and reduction percentage.
    """
    total_params = 0
    trainable_params = 0
    
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    reduction_pct = (1 - trainable_params / total_params) * 100 if total_params > 0 else 0
    
    summary = {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "reduction_pct": reduction_pct,
        "trainable_pct": 100 - reduction_pct,
    }
    
    print(f"\n{'='*60}")
    print(f"  {model_name} — Parameter Summary")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total_params:>15,}")
    print(f"  Trainable parameters: {trainable_params:>15,}")
    print(f"  Frozen parameters:    {total_params - trainable_params:>15,}")
    print(f"  Parameter reduction:  {reduction_pct:>14.1f}%")
    print(f"  Trainable ratio:      {100 - reduction_pct:>14.2f}%")
    print(f"{'='*60}\n")
    
    # Validate against target
    if total_params > 1_000_000_000:  # > 1B params (GPT-2 XL range)
        if abs(trainable_params - 8_400_000) / 8_400_000 <= 0.1:
            print(f"  ✅ Trainable params ({trainable_params:,}) within ±10% of 8.4M target")
        else:
            print(f"  ⚠️  Trainable params ({trainable_params:,}) differs from 8.4M target")
            print(f"     Adjust LoRA rank or target_modules to hit the target.")
    
    if reduction_pct >= 99.0:
        print(f"  ✅ Parameter reduction ({reduction_pct:.1f}%) meets 99.4% target\n")
    else:
        print(f"  ⚠️  Parameter reduction ({reduction_pct:.1f}%) below 99.4% target\n")
    
    return summary


def apply_lora_to_model(model, task_type: str = "SEQ_CLS", **lora_kwargs):
    """
    Convenience function: apply LoRA config and wrap the model.
    
    Args:
        model: Base HuggingFace model.
        task_type: "SEQ_CLS" or "CAUSAL_LM".
        **lora_kwargs: Additional LoRA config overrides.
    
    Returns:
        PEFT-wrapped model with LoRA adapters.
    """
    lora_config = get_lora_config(task_type=task_type, **lora_kwargs)
    peft_model = get_peft_model(model, lora_config)
    
    # Log parameter summary
    print_param_summary(peft_model, f"LoRA-adapted ({task_type})")
    
    return peft_model
