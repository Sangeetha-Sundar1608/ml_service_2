"""
ML Service Configuration
All environment variables and settings centralized here
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # ============================================
    # SERVICE CONFIGURATION
    # ============================================
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = True
    ML_SERVICE_PORT: int = 8001
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # ============================================
    # LLM PROVIDERS
    # ============================================
    # vLLM (GPU - local inference)
    VLLM_SERVICE_URL: Optional[str] = None  # e.g., "http://vllm-service:8000"
    VLLM_GRPC_URL: Optional[str] = None     # e.g., "vllm-service:9000" (Phase 2)
    VLLM_TIMEOUT: int = 120                 # seconds
    
    # Ollama (CPU - fallback)
    OLLAMA_URL: Optional[str] = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 60
    
    # OpenAI (API - expensive fallback)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_TIMEOUT: int = 60
    OPENAI_MAX_RETRIES: int = 3
    
    # Anthropic (API - expensive fallback)
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_TIMEOUT: int = 60
    ANTHROPIC_MAX_RETRIES: int = 3
    
    # ============================================
    # CIRCUIT BREAKER
    # ============================================
    CIRCUIT_FAILURE_THRESHOLD: int = 5     # Open circuit after N failures
    CIRCUIT_TIMEOUT: int = 300              # Seconds before retry (5 min)
    HEALTH_CHECK_CACHE_TTL: int = 30        # Cache health status (seconds)
    
    # ============================================
    # DATABASE
    # ============================================
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/ml_service"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    
    # ============================================
    # REDIS (Cache + Celery)
    # ============================================
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600             # 1 hour
    REDIS_JOB_RESULT_TTL: int = 86400       # 24 hours
    
    # ============================================
    # CELERY (Async Tasks)
    # ============================================
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    
    # ============================================
    # EMBEDDING SERVICE
    # ============================================
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_CACHE_ENABLED: bool = True
    
    # ============================================
    # DOCUMENT PROCESSING
    # ============================================
    MAX_FILE_SIZE_MB: int = 50               # PDF, DOCX, images
    MAX_AUDIO_SIZE_MB: int = 500             # Audio files
    MAX_VIDEO_SIZE_MB: int = 1000            # Video files
    
    TESSERACT_CMD: Optional[str] = None      # Path to tesseract (OCR)
    WHISPER_MODEL: str = "base"              # Whisper model size
    
    # ============================================
    # COST TRACKING & ALERTS
    # ============================================
    OPENAI_DAILY_COST_ALERT: float = 50.0   # Alert if > $50/day
    EMBEDDING_DAILY_COST_ALERT: float = 10.0  # Alert if > $10/day
    ERROR_RATE_ALERT_THRESHOLD: float = 0.05  # 5%
    LATENCY_P95_ALERT_MS: int = 2000         # 2 seconds
    
    # ============================================
    # PERFORMANCE
    # ============================================
    LLM_REQUEST_TIMEOUT: int = 120           # Max time for LLM request
    EMBEDDING_REQUEST_TIMEOUT: int = 30
    DOCUMENT_PROCESSING_TIMEOUT: int = 60
    
    MAX_CONCURRENT_REQUESTS: int = 100
    CONNECTION_POOL_SIZE: int = 50
    
    # ============================================
    # MODEL MANAGEMENT
    # ============================================
    MAX_LOADED_MODELS: int = 3               # Keep N models in memory
    GPU_MEMORY_THRESHOLD_GB: int = 25        # Unload if > threshold
    CPU_MEMORY_THRESHOLD_PERCENT: int = 85
    
    # ============================================
    # CORS
    # ============================================
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",             # Platform Service (local)
        "http://localhost:3000",             # Frontend (local)
    ]
    
    # ============================================
    # MONITORING
    # ============================================
    PROMETHEUS_PORT: int = 9090
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = False             # OpenTelemetry (future)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Validation on startup
def validate_settings():
    """Validate critical settings on startup"""
    errors = []
    
    # At least one LLM provider must be configured
    if not any([
        settings.VLLM_SERVICE_URL,
        settings.OLLAMA_URL,
        settings.OPENAI_API_KEY,
        settings.ANTHROPIC_API_KEY
    ]):
        errors.append("At least one LLM provider must be configured")
    
    # Database required
    if not settings.DATABASE_URL:
        errors.append("DATABASE_URL is required")
    
    # Redis required
    if not settings.REDIS_URL:
        errors.append("REDIS_URL is required")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")


# Run validation
validate_settings()