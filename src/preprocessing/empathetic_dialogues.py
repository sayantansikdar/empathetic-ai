"""
EmpatheticDialogues → prompt/response pairs for the response generator.

Each EmpatheticDialogues conversation opens with a speaker describing a
situation and a crowd-worker "listener" replying empathetically. We train on
that first exchange: the speaker's message (conditioned on a 3-class sentiment
derived from the conversation's emotion label) → the listener's real reply.

Source: Rashkin et al., 2019, via the Parquet copy `Estwld/empathetic_dialogues_llm`
(the original `facebook/empathetic_dialogues` uses a loading script that
current `datasets` releases no longer execute).
"""

import logging
import re
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

ED_DATASET = "Estwld/empathetic_dialogues_llm"

# The 32 EmpatheticDialogues emotions mapped onto the classifier's 3 classes,
# so the generator is conditioned on the same labels the classifier predicts.
ED_EMOTION_TO_SENTIMENT = {
    # positive
    "anticipating": "positive", "caring": "positive", "confident": "positive",
    "content": "positive", "excited": "positive", "faithful": "positive",
    "grateful": "positive", "hopeful": "positive", "impressed": "positive",
    "joyful": "positive", "prepared": "positive", "proud": "positive",
    "surprised": "positive", "trusting": "positive",
    # negative
    "afraid": "negative", "angry": "negative", "annoyed": "negative",
    "anxious": "negative", "apprehensive": "negative", "ashamed": "negative",
    "devastated": "negative", "disappointed": "negative", "disgusted": "negative",
    "embarrassed": "negative", "furious": "negative", "guilty": "negative",
    "jealous": "negative", "lonely": "negative", "sad": "negative",
    "terrified": "negative",
    # mixed / reflective
    "nostalgic": "neutral", "sentimental": "neutral",
}

_SPACES = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = text.replace("_comma_", ",")
    return _SPACES.sub(" ", text).strip()


def _first_exchange(example: dict) -> dict:
    """Pull the opening speaker message and the listener's first reply."""
    turns = example.get("conversations") or []
    user = next((t["content"] for t in turns if t.get("role") == "user"), "")
    reply = ""
    seen_user = False
    for t in turns:
        if t.get("role") == "user":
            seen_user = True
        elif t.get("role") == "assistant" and seen_user:
            reply = t["content"]
            break
    return {
        "user_text": _clean(user),
        "response": _clean(reply),
        "emotion": ED_EMOTION_TO_SENTIMENT.get(example.get("emotion", ""), "neutral"),
        "fine_emotion": example.get("emotion", ""),
    }


def load_empathetic_pairs(
    cache_dir: str = "./data",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> DatasetDict:
    """Load EmpatheticDialogues as (user_text, emotion, response) pairs."""
    raw = load_dataset(ED_DATASET, cache_dir=cache_dir)
    pairs = DatasetDict()
    for split, target in (("train", "train"), ("valid", "validation"), ("test", "test")):
        if split not in raw:
            continue
        ds = raw[split].map(_first_exchange, remove_columns=raw[split].column_names)
        ds = ds.filter(lambda ex: len(ex["user_text"].split()) >= 3 and len(ex["response"].split()) >= 3)
        cap = max_train_samples if target == "train" else max_eval_samples
        if cap:
            ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
        pairs[target] = ds
    logger.info("EmpatheticDialogues pairs — " + ", ".join(f"{k}: {len(v)}" for k, v in pairs.items()))
    return pairs


def tokenize_pairs(dataset: Dataset, tokenizer, build_prompt, max_length: int = 128) -> Dataset:
    """
    Tokenize prompt + response for causal-LM training.

    Loss is computed on the listener's reply (and the closing EOS) only:
    prompt tokens are masked with -100, so the model learns to *respond*,
    not to reproduce the user's message or the template.
    """
    eos = tokenizer.eos_token_id

    def encode(ex):
        prompt_ids = tokenizer(build_prompt(ex["user_text"], ex["emotion"]), add_special_tokens=False)["input_ids"]
        reply_ids = tokenizer(" " + ex["response"], add_special_tokens=False)["input_ids"] + [eos]
        # Keep the reply intact; trim the prompt from the left if needed.
        budget = max_length - len(reply_ids)
        if budget < 8:
            reply_ids = reply_ids[: max_length - 8]
            budget = 8
        prompt_ids = prompt_ids[-budget:]
        input_ids = prompt_ids + reply_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + reply_ids,
        }

    return dataset.map(encode, remove_columns=dataset.column_names)
