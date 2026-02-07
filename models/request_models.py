"""
Request Models - Pydantic schemas for API requests
Validates incoming data to ML Service endpoints
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


# ============================================
# EPIC 1: LLM Inference Requests
# ============================================

class LLMMessage(BaseModel):
    """Single message in a conversation"""
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('Role must be system, user, or assistant')
        return v


class LLMInferenceRequest(BaseModel):
    """Request for LLM inference"""
    messages: List[LLMMessage] = Field(..., description="Conversation messages")
    model: Optional[str] = Field(None, description="Model name (uses default if not provided)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stream: bool = Field(False, description="Stream response token-by-token")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences to stop generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "temperature": 0.7,
                "max_tokens": 256
            }
        }


# ============================================
# EPIC 2: Document Processing Requests
# ============================================

class DocumentFormat(str, Enum):
    """Supported document formats"""
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class DocumentParseRequest(BaseModel):
    """Request for document parsing (file uploaded separately)"""
    format: Optional[DocumentFormat] = Field(None, description="Document format (auto-detected if not provided)")
    extract_metadata: bool = Field(True, description="Extract metadata from document")
    ocr_enabled: bool = Field(True, description="Use OCR for scanned documents/images")
    language: Optional[str] = Field("en", description="Document language (auto-detect if None)")


# ============================================
# EPIC 3: Embedding Requests
# ============================================

class EmbeddingRequest(BaseModel):
    """Request for text embeddings"""
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="Texts to embed")
    model: Optional[str] = Field("all-MiniLM-L6-v2", description="Embedding model")
    normalize: bool = Field(True, description="L2 normalize embeddings")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not all(text.strip() for text in v):
            raise ValueError('All texts must be non-empty')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello world", "Machine learning is awesome"],
                "normalize": True
            }
        }


class BatchEmbeddingRequest(BaseModel):
    """Request for batch embeddings (async processing)"""
    texts: List[str] = Field(..., min_items=1, max_items=10000, description="Texts to embed")
    model: Optional[str] = Field("all-MiniLM-L6-v2", description="Embedding model")
    webhook_url: Optional[str] = Field(None, description="Callback URL for completion")
    callback_headers: Optional[Dict[str, str]] = Field(None, description="Headers for callback")


# ============================================
# EPIC 4: Job Management Requests
# ============================================

class JobStatus(str, Enum):
    """Job status values"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TimeRange(str, Enum):
    """Time range for analytics"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class AnalyticsRequest(BaseModel):
    """Request for analytics data"""
    time_range: TimeRange = Field(TimeRange.DAY, description="Time range for analytics")
    provider: Optional[str] = Field(None, description="Filter by provider (vllm, ollama, openai, anthropic)")
    model: Optional[str] = Field(None, description="Filter by model name")