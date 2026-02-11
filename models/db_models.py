"""
Database Models - SQLAlchemy ORM models
Defines database tables for jobs, costs, and usage tracking
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()


# ============================================
# EPIC 4: Job Tracking
# ============================================

class JobStatusEnum(str, enum.Enum):
    """Job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncJob(Base):
    """
    Tracks async jobs (batch embeddings, document processing)
    EPIC 4.2: Async Job Handling
    """
    __tablename__ = "async_jobs"
    
    id = Column(String(36), primary_key=True)  # UUID
    job_type = Column(String(50), nullable=False, index=True)  # 'embedding', 'document', etc.
    status = Column(SQLEnum(JobStatusEnum), default=JobStatusEnum.QUEUED, nullable=False, index=True)
    
    # Request data
    request_data = Column(JSON, nullable=False)  # Original request
    
    # Progress tracking
    progress_percent = Column(Integer, default=0)
    total_items = Column(Integer)
    processed_items = Column(Integer, default=0)
    
    # Result data
    result = Column(JSON, nullable=True)  # Result when completed
    error_message = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Callback info
    webhook_url = Column(String(500), nullable=True)
    callback_headers = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<AsyncJob(id={self.id}, type={self.job_type}, status={self.status})>"


# ============================================
# EPIC 4: Cost Tracking
# ============================================

class ProviderEnum(str, enum.Enum):
    """LLM provider enumeration"""
    VLLM = "vllm"
    VLLM_GRPC = "vllm_grpc"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ProviderUsageLog(Base):
    """
    Logs each provider request for cost tracking and analytics
    EPIC 4.3: Cost Tracking & Analytics
    """
    __tablename__ = "provider_usage_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Provider info
    provider = Column(SQLEnum(ProviderEnum), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    
    # Request details
    request_type = Column(String(50), nullable=False)  # 'chat', 'completion', 'embedding'
    
    # Token usage
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, nullable=False)
    
    # Performance
    latency_ms = Column(Integer, nullable=False)
    success = Column(Boolean, default=True, nullable=False, index=True)
    
    # Cost (for API providers)
    cost_usd = Column(Float, default=0.0)
    
    # Timestamp
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    # Optional metadata
    error_message = Column(Text, nullable=True)
    request_id = Column(String(100), nullable=True, index=True)
    
    def __repr__(self):
        return f"<ProviderUsage(provider={self.provider}, model={self.model}, tokens={self.total_tokens})>"


class DailyCostSummary(Base):
    """
    Daily aggregated cost summary (for alerts)
    EPIC 4.3: Cost Tracking & Analytics
    """
    __tablename__ = "daily_cost_summary"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)  # Date (midnight UTC)
    
    # Per-provider costs
    vllm_requests = Column(Integer, default=0)
    vllm_tokens = Column(Integer, default=0)
    
    ollama_requests = Column(Integer, default=0)
    ollama_tokens = Column(Integer, default=0)
    
    openai_requests = Column(Integer, default=0)
    openai_tokens = Column(Integer, default=0)
    openai_cost_usd = Column(Float, default=0.0)
    
    anthropic_requests = Column(Integer, default=0)
    anthropic_tokens = Column(Integer, default=0)
    anthropic_cost_usd = Column(Float, default=0.0)
    
    # Totals
    total_requests = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    total_cost_usd = Column(Float, nullable=False)
    
    # Quality metrics
    error_count = Column(Integer, default=0)
    avg_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<DailyCostSummary(date={self.date}, cost=${self.total_cost_usd:.4f})>"


# ============================================
# Model Metadata (Optional - for ML-1.4)
# ============================================

class LoadedModel(Base):
    """
    Tracks currently loaded models in memory
    EPIC 1.4: Model Management
    """
    __tablename__ = "loaded_models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(200), nullable=False, unique=True, index=True)
    provider = Column(SQLEnum(ProviderEnum), nullable=False)
    
    # Memory usage
    memory_mb = Column(Float, nullable=False)
    
    # Usage tracking
    last_used_at = Column(DateTime, default=func.now(), nullable=False)
    total_requests = Column(Integer, default=0)
    
    # Status
    is_loaded = Column(Boolean, default=True)
    loaded_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<LoadedModel(name={self.model_name}, memory={self.memory_mb}MB)>"
    