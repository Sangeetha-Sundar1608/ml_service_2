"""
Response Models - Pydantic schemas for API responses
Standardizes output from ML Service endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from models.request_models import JobStatus


# ============================================
# EPIC 1: LLM Inference Responses
# ============================================

class LLMInferenceResponse(BaseModel):
    """Response from LLM inference"""
    content: str = Field(..., description="Generated text")
    tokens_used: int = Field(..., description="Total tokens used")
    model: str = Field(..., description="Model used for inference")
    provider: str = Field(..., description="Provider: vllm, ollama, openai, anthropic")
    finish_reason: str = Field("stop", description="Reason generation stopped")
    latency_ms: Optional[int] = Field(None, description="Request latency in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Hello! I'm doing well, thank you for asking.",
                "tokens_used": 15,
                "model": "meta-llama/Llama-3.2-3B",
                "provider": "vllm",
                "finish_reason": "stop",
                "latency_ms": 150
            }
        }


# ============================================
# EPIC 2: Document Processing Responses
# ============================================

class DocumentMetadata(BaseModel):
    """Document metadata"""
    created_date: Optional[str] = None
    author: Optional[str] = None
    encoding: Optional[str] = None
    language: Optional[str] = None


class DocumentParseResponse(BaseModel):
    """Response from document parsing"""
    text: str = Field(..., description="Extracted text")
    format: str = Field(..., description="Document format")
    pages: Optional[int] = Field(None, description="Number of pages (PDF only)")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Extraction quality (0-1)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="OCR confidence (images only)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Resume content here...",
                "format": "pdf",
                "pages": 2,
                "quality_score": 0.95,
                "processing_time_ms": 1200,
                "metadata": {
                    "created_date": "2024-01-15",
                    "encoding": "utf-8",
                    "language": "en"
                }
            }
        }


# ============================================
# EPIC 3: Embedding Responses
# ============================================

class EmbeddingResponse(BaseModel):
    """Response from embedding generation"""
    embeddings: List[List[float]] = Field(..., description="384-dimensional vectors")
    model: str = Field(..., description="Model used")
    provider: str = Field("local", description="Provider: local or openai")
    tokens: int = Field(..., description="Total tokens processed")
    dimension: int = Field(384, description="Embedding dimension")
    processing_time_ms: int = Field(..., description="Processing time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [[0.1, 0.2, 0.3]],  # Truncated for example
                "model": "all-MiniLM-L6-v2",
                "provider": "local",
                "tokens": 10,
                "dimension": 384,
                "processing_time_ms": 50
            }
        }


class BatchEmbeddingResponse(BaseModel):
    """Response from batch embedding request"""
    job_id: str = Field(..., description="Job ID for tracking")
    status: JobStatus = Field(..., description="Job status")
    estimated_time_seconds: Optional[int] = Field(None, description="Estimated completion time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "estimated_time_seconds": 30
            }
        }


# ============================================
# EPIC 4: Health & Analytics Responses
# ============================================

class ProviderHealth(BaseModel):
    """Health status of a provider"""
    status: str = Field(..., description="healthy, degraded, or unhealthy")
    latency_ms: Optional[int] = Field(None, description="Health check latency")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class ResourceStatus(BaseModel):
    """System resource status"""
    gpu_available: bool = Field(..., description="GPU available")
    gpu_memory_gb: Optional[float] = Field(None, description="GPU memory used")
    gpu_memory_total_gb: Optional[float] = Field(None, description="Total GPU memory")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_gb: float = Field(..., description="RAM used")
    memory_total_gb: float = Field(..., description="Total RAM")


class HealthResponse(BaseModel):
    """ML Service health check response"""
    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, Any] = Field(..., description="Status of each service")
    resources: ResourceStatus = Field(..., description="System resources")
    uptime_seconds: int = Field(..., description="Service uptime")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-15T10:30:00Z",
                "services": {
                    "llm_inference": {
                        "vllm": {"status": "healthy", "latency_ms": 5},
                        "ollama": {"status": "healthy", "latency_ms": 100}
                    },
                    "embeddings": {"status": "healthy"},
                    "document_processing": {"status": "healthy"}
                },
                "resources": {
                    "gpu_available": True,
                    "gpu_memory_gb": 20.5,
                    "gpu_memory_total_gb": 24.0,
                    "cpu_usage_percent": 45.2,
                    "memory_usage_gb": 8.1,
                    "memory_total_gb": 32.0
                },
                "uptime_seconds": 86400
            }
        }


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Current status")
    progress_percent: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    result: Optional[Any] = Field(None, description="Result when completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")


class ProviderUsage(BaseModel):
    """Provider usage statistics"""
    provider: str = Field(..., description="Provider name")
    requests_total: int = Field(..., description="Total requests")
    requests_success: int = Field(..., description="Successful requests")
    requests_failed: int = Field(..., description="Failed requests")
    tokens_total: int = Field(..., description="Total tokens processed")
    latency_avg_ms: float = Field(..., description="Average latency")
    latency_p95_ms: float = Field(..., description="P95 latency")
    cost_usd: float = Field(0.0, description="Estimated cost in USD")


class AnalyticsResponse(BaseModel):
    """Analytics response"""
    time_range: str = Field(..., description="Time range queried")
    providers: List[ProviderUsage] = Field(..., description="Per-provider statistics")
    total_requests: int = Field(..., description="Total requests across all providers")
    total_cost_usd: float = Field(..., description="Total estimated cost")
    error_rate_percent: float = Field(..., description="Overall error rate")
    