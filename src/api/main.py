"""
EmpaChat FastAPI Service

Serves emotion classification and empathetic response generation.

Start: uvicorn src.api.main:app --reload --port 8000
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.schemas import (
    EmotionRequest,
    EmotionResponse,
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
)
from src.preprocessing.text_cleaner import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Global model instances (loaded at startup)
# ──────────────────────────────────────────────────────────────────────────────

_classifier = None
_generator = None
_model_name = os.getenv("EMPACHAT_MODEL", "gpt2-xl")
_device = None


def _load_classifier():
    """Lazy-load the emotion classifier."""
    global _classifier, _device
    
    from src.models.emotion_classifier import EmotionClassifier
    
    adapter_path = os.getenv(
        "EMPACHAT_CLASSIFIER_ADAPTER",
        "./outputs/classifier/classifier_lora",
    )
    
    _classifier = EmotionClassifier(
        model_name=_model_name,
        num_classes=3,
        use_lora=True,
        lora_adapter_path=adapter_path if Path(adapter_path).exists() else None,
    )
    _classifier.load_model()
    _device = _classifier.device
    logger.info(f"Classifier loaded on {_device}")


def _load_generator():
    """Lazy-load the response generator."""
    global _generator
    
    from src.models.response_generator import ResponseGenerator
    
    adapter_path = os.getenv(
        "EMPACHAT_GENERATOR_ADAPTER",
        "./outputs/generator/generator_lora",
    )
    
    _generator = ResponseGenerator(
        model_name=_model_name,
        use_lora=True,
        lora_adapter_path=adapter_path if Path(adapter_path).exists() else None,
    )
    _generator.load_model()
    logger.info("Generator loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("EmpaChat API starting up...")
    
    try:
        _load_classifier()
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        logger.warning("Classifier will not be available")
    
    try:
        _load_generator()
    except Exception as e:
        logger.error(f"Failed to load generator: {e}")
        logger.warning("Generator will not be available")
    
    logger.info("EmpaChat API ready!")
    yield
    
    logger.info("EmpaChat API shutting down...")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EmpaChat API",
    description="Empathetic Response AI — Emotion classification and empathetic response generation powered by GPT-2 XL with LoRA fine-tuning.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check — returns service status and model availability."""
    return HealthResponse(
        status="healthy",
        classifier_loaded=_classifier is not None,
        generator_loaded=_generator is not None,
        model_name=_model_name,
        device=_device or "unknown",
    )


@app.post("/classify_emotion", response_model=EmotionResponse, tags=["Classification"])
async def classify_emotion(request: EmotionRequest):
    """
    Classify the emotion of the input text.
    
    Returns the predicted emotion label and confidence scores for all classes.
    """
    if _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier model not loaded. Please restart the service.",
        )
    
    # Clean input
    cleaned_text = clean_text(request.text)
    
    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Input text is empty after cleaning.",
        )
    
    try:
        start = time.time()
        emotion, scores = _classifier.predict(cleaned_text)
        latency = time.time() - start
        
        logger.info(
            f"Classified: '{cleaned_text[:50]}...' → {emotion} "
            f"({max(scores.values()):.3f}) [{latency*1000:.0f}ms]"
        )
        
        return EmotionResponse(
            emotion=emotion,
            confidence=max(scores.values()),
            all_scores=scores,
            text=request.text,
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_response", response_model=GenerationResponse, tags=["Generation"])
async def generate_response(request: GenerationRequest):
    """
    Generate an empathetic response to the user's message.
    
    Automatically detects emotion if not provided, then generates
    a contextually appropriate empathetic reply.
    """
    # Clean input
    cleaned_text = clean_text(request.text)
    
    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Input text is empty after cleaning.",
        )
    
    # Detect emotion
    emotion = request.emotion
    emotion_confidence = 1.0
    
    if emotion is None:
        if _classifier is None:
            emotion = "neutral"
            emotion_confidence = 0.0
            logger.warning("Classifier not loaded, defaulting to 'neutral'")
        else:
            try:
                emotion, scores = _classifier.predict(cleaned_text)
                emotion_confidence = max(scores.values())
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                emotion = "neutral"
                emotion_confidence = 0.0
    
    # Generate response
    if _generator is not None:
        try:
            start = time.time()
            response_text = _generator.generate_response(
                cleaned_text,
                emotion,
                temperature=request.temperature,
                max_new_tokens=request.max_tokens,
            )
            latency = time.time() - start
            
            logger.info(
                f"Generated response for emotion={emotion} [{latency*1000:.0f}ms]"
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            response_text = _get_fallback_response(emotion)
    else:
        response_text = _get_fallback_response(emotion)
        logger.warning("Generator not loaded, using fallback response")
    
    return GenerationResponse(
        response=response_text,
        detected_emotion=emotion,
        emotion_confidence=emotion_confidence,
        text=request.text,
    )


def _get_fallback_response(emotion: str) -> str:
    """Provide a fallback response when the generator is unavailable."""
    fallbacks = {
        "positive": "That's wonderful to hear! Your positive energy is truly uplifting. I hope you continue to experience these beautiful moments.",
        "negative": "I'm sorry you're going through this. Your feelings are completely valid, and I want you to know that you're not alone. Things will get better.",
        "neutral": "Thank you for sharing that with me. I appreciate you opening up, and I'm here if you'd like to talk more about it.",
    }
    return fallbacks.get(emotion, fallbacks["neutral"])


# ──────────────────────────────────────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
