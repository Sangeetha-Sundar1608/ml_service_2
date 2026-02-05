
"""
Shared dependency container and injection helpers
"""
from services.llm_router import LLMRouter
from services.embedding_service import EmbeddingService
from services.document_processor import DocumentProcessor
from services.health_check import HealthChecker
from services.cost_tracker import CostTracker

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import settings

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global service instances
class ServiceContainer:
    """Shared service instances across the application"""
    llm_router: LLMRouter = None
    embedding_service: EmbeddingService = None
    document_processor: DocumentProcessor = None
    health_checker: HealthChecker = None
    cost_tracker: CostTracker = None


services = ServiceContainer()


# Dependency injection helpers
def get_llm_router() -> LLMRouter:
    """Get LLM router instance"""
    return services.llm_router


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance"""
    return services.embedding_service


def get_document_processor() -> DocumentProcessor:
    """Get document processor instance"""
    return services.document_processor


def get_health_checker() -> HealthChecker:
    """Get health checker instance"""
    return services.health_checker


def get_cost_tracker() -> CostTracker:
    """Get cost tracker instance"""
    return services.cost_tracker
