"""
Text cleaning utilities and emotion label mappings for GoEmotions dataset.
"""

import re
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# GoEmotions 27+1 Emotion Labels (index → label)
# ──────────────────────────────────────────────────────────────────────────────

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}


# ──────────────────────────────────────────────────────────────────────────────
# Simplified Sentiment Mapping (27 → 3 classes)
# Standard grouping from Demszky et al., 2020
# ──────────────────────────────────────────────────────────────────────────────

SENTIMENT_LABELS = ["positive", "negative", "neutral"]

EMOTION_TO_SENTIMENT = {
    # Positive emotions
    "admiration": "positive",
    "amusement": "positive",
    "approval": "positive",
    "caring": "positive",
    "desire": "positive",
    "excitement": "positive",
    "gratitude": "positive",
    "joy": "positive",
    "love": "positive",
    "optimism": "positive",
    "pride": "positive",
    "relief": "positive",
    "surprise": "positive",
    # Negative emotions
    "anger": "negative",
    "annoyance": "negative",
    "disappointment": "negative",
    "disapproval": "negative",
    "disgust": "negative",
    "embarrassment": "negative",
    "fear": "negative",
    "grief": "negative",
    "nervousness": "negative",
    "remorse": "negative",
    "sadness": "negative",
    # Neutral / Ambiguous
    "confusion": "neutral",
    "curiosity": "neutral",
    "realization": "neutral",
    "neutral": "neutral",
}

SENTIMENT_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT = {idx: label for idx, label in enumerate(SENTIMENT_LABELS)}


# ──────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────────────────────

# Regex patterns compiled once for performance
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_USERNAME_PATTERN = re.compile(r"@\w+")
_SUBREDDIT_PATTERN = re.compile(r"r/\w+")
_HTML_ENTITY_PATTERN = re.compile(r"&\w+;")
_EXTRA_WHITESPACE_PATTERN = re.compile(r"\s+")
_SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s.,!?;:'\"-]")


def clean_text(text: str, lowercase: bool = True, 
               remove_urls: bool = True,
               remove_usernames: bool = True,
               remove_special: bool = False) -> str:
    """
    Clean input text for emotion classification.
    
    Args:
        text: Raw input text.
        lowercase: Convert to lowercase.
        remove_urls: Strip URLs.
        remove_usernames: Strip @mentions.
        remove_special: Remove non-standard characters (keeps emojis if False).
    
    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs
    if remove_urls:
        text = _URL_PATTERN.sub(" ", text)
    
    # Remove Reddit usernames
    if remove_usernames:
        text = _USERNAME_PATTERN.sub(" ", text)
    
    # Remove subreddit references
    text = _SUBREDDIT_PATTERN.sub(" ", text)
    
    # Remove HTML entities
    text = _HTML_ENTITY_PATTERN.sub(" ", text)
    
    # Remove special characters (but keep basic punctuation and emojis)
    if remove_special:
        text = _SPECIAL_CHARS_PATTERN.sub(" ", text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Normalize whitespace
    text = _EXTRA_WHITESPACE_PATTERN.sub(" ", text).strip()
    
    return text


def get_num_labels(num_classes: int = 3) -> int:
    """Get the number of classification labels."""
    if num_classes == 3:
        return len(SENTIMENT_LABELS)
    elif num_classes == 27:
        return len(EMOTION_LABELS) - 1  # Exclude 'neutral' from 27 fine-grained
    elif num_classes == 28:
        return len(EMOTION_LABELS)
    else:
        raise ValueError(f"num_classes must be 3, 27, or 28, got {num_classes}")


def get_label_names(num_classes: int = 3) -> list[str]:
    """Get label names for the specified classification scheme."""
    if num_classes == 3:
        return SENTIMENT_LABELS
    elif num_classes == 27:
        return EMOTION_LABELS[:-1]  # Exclude 'neutral'
    elif num_classes == 28:
        return EMOTION_LABELS
    else:
        raise ValueError(f"num_classes must be 3, 27, or 28, got {num_classes}")


def map_emotion_to_sentiment(emotion_label: str) -> Optional[str]:
    """Map a fine-grained emotion label to a sentiment category."""
    return EMOTION_TO_SENTIMENT.get(emotion_label)


def map_emotion_id_to_sentiment_id(emotion_id: int) -> int:
    """Map a fine-grained emotion ID to a sentiment class ID."""
    emotion_label = ID_TO_LABEL.get(emotion_id, "neutral")
    sentiment_label = EMOTION_TO_SENTIMENT.get(emotion_label, "neutral")
    return SENTIMENT_TO_ID[sentiment_label]
