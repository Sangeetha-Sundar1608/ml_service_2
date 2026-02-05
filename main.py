"""
ML Service - FastAPI Application Entry Point
Handles LLM inference, document processing, embeddings with intelligent routing
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
from datetime import datetime

from config import settings
from routes import llm, embeddings, documents, health, analytics
from services.llm_router import LLMRouter
from services.embedding_service import EmbeddingService
from services.document_processor import DocumentProcessor
from services.health_check import HealthChecker
from services.cost_tracker import CostTracker

# Configure structured JSON logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


from dependencies import services


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("🚀 ML Service starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Port: {settings.ML_SERVICE_PORT}")
    
    # Initialize services
    try:
        services.llm_router = LLMRouter()
        services.embedding_service = EmbeddingService()
        services.document_processor = DocumentProcessor()
        services.health_checker = HealthChecker()
        services.cost_tracker = CostTracker()
        
        logger.info("✅ All services initialized successfully")
        
        # Log provider configuration
        logger.info(f"vLLM URL: {settings.VLLM_SERVICE_URL or 'Not configured'}")
        logger.info(f"Ollama URL: {settings.OLLAMA_URL or 'Not configured'}")
        logger.info(f"OpenAI: {'Configured' if settings.OPENAI_API_KEY else 'Not configured'}")
        logger.info(f"Anthropic: {'Configured' if settings.ANTHROPIC_API_KEY else 'Not configured'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 ML Service shutting down...")
    # Cleanup resources if needed
    logger.info("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ML Service",
    description="Distributed inference + data processing platform",
    version="3.0.0",
    lifespan=lifespan
)


# CORS middleware (for Platform Service)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Custom exception for all providers failed
class AllProvidersFailedError(Exception):
    """Raised when all LLM providers are unavailable"""
    pass


@app.exception_handler(AllProvidersFailedError)
async def all_providers_failed_handler(request: Request, exc: AllProvidersFailedError):
    """Handle case where all LLM providers failed"""
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service unavailable",
            "message": "All LLM providers are currently unavailable. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Include routers
app.include_router(health.router, prefix="/ml/v1", tags=["Health"])
app.include_router(llm.router, prefix="/ml/v1/llm", tags=["LLM Inference"])
app.include_router(embeddings.router, prefix="/ml/v1/embeddings", tags=["Embeddings"])
app.include_router(documents.router, prefix="/ml/v1/parse", tags=["Document Processing"])
app.include_router(analytics.router, prefix="/ml/v1/analytics", tags=["Analytics"])


# Root endpoint
@app.get("/")
async def root():
    """Service info"""
    return {
        "service": "ML Service",
        "version": "3.0.1",
        "status": "running - 3.0.1",
        "endpoints": {
            "health": "/ml/v1/health",
            "llm": "/ml/v1/llm/*",
            "embeddings": "/ml/v1/embeddings/*",
            "documents": "/ml/v1/parse/*",
            "analytics": "/ml/v1/analytics/*",
        }
    }





if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.ML_SERVICE_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )