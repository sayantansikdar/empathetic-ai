"""
Pydantic schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class EmotionRequest(BaseModel):
    """Request body for /classify_emotion endpoint."""
    text: str = Field(
        ...,
        description="User message to classify",
        min_length=1,
        max_length=2000,
        examples=["I'm feeling so happy today!"],
    )


class EmotionResponse(BaseModel):
    """Response from /classify_emotion endpoint."""
    emotion: str = Field(
        ..., description="Predicted emotion label"
    )
    confidence: float = Field(
        ..., description="Confidence score for the predicted emotion"
    )
    all_scores: dict[str, float] = Field(
        ..., description="Confidence scores for all classes"
    )
    text: str = Field(
        ..., description="Original input text"
    )


class GenerationRequest(BaseModel):
    """Request body for /generate_response endpoint."""
    text: str = Field(
        ...,
        description="User message to respond to",
        min_length=1,
        max_length=2000,
        examples=["I just lost my job and feel terrible."],
    )
    emotion: Optional[str] = Field(
        None,
        description="Override emotion label (auto-detected if not provided)",
    )
    temperature: Optional[float] = Field(
        None,
        description="Generation temperature (0.1-2.0)",
        ge=0.1,
        le=2.0,
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum tokens to generate",
        ge=10,
        le=500,
    )


class GenerationResponse(BaseModel):
    """Response from /generate_response endpoint."""
    response: str = Field(
        ..., description="Generated empathetic response"
    )
    detected_emotion: str = Field(
        ..., description="Detected emotion from input"
    )
    emotion_confidence: float = Field(
        ..., description="Confidence in emotion detection"
    )
    text: str = Field(
        ..., description="Original input text"
    )


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str = Field(..., description="Service status")
    classifier_loaded: bool = Field(
        ..., description="Whether classifier model is loaded"
    )
    generator_loaded: bool = Field(
        ..., description="Whether generator model is loaded"
    )
    model_name: str = Field(
        ..., description="Base model name"
    )
    device: str = Field(
        ..., description="Compute device (cuda/cpu)"
    )
