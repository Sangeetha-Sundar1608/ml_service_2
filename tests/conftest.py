"""
Pytest configuration and shared test fixtures
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app, get_db
from models.db_models import Base


# ============================================
# DATABASE FIXTURES
# ============================================

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine (in-memory SQLite)"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_db(test_engine):
    """Create test database session"""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine
    )
    
    db = TestingSessionLocal()
    
    yield db
    
    # Rollback after each test
    db.rollback()
    db.close()


@pytest.fixture(scope="function")
def client(test_db):
    """FastAPI test client with test database"""
    # Override database dependency
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    # Create test client
    with TestClient(app) as c:
        yield c
    
    # Cleanup
    app.dependency_overrides.clear()


# ============================================
# MOCK DATA FIXTURES
# ============================================

@pytest.fixture
def sample_llm_request():
    """Sample LLM request for testing"""
    return {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }


@pytest.fixture
def sample_embedding_request():
    """Sample embedding request"""
    return {
        "texts": ["Hello world", "Machine learning is awesome"],
        "normalize": True
    }


@pytest.fixture
def sample_document_file(tmp_path):
    """Create sample PDF file for testing"""
    pdf_path = tmp_path / "test.pdf"
    
    # Create minimal PDF
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Test Document")
    c.drawString(100, 700, "This is a test PDF file.")
    c.save()
    
    return pdf_path


# ============================================
# MOCK SERVICES
# ============================================

@pytest.fixture
def mock_vllm_client(monkeypatch):
    """Mock vLLM client (no real GPU needed for tests)"""
    from clients.vllm_client import VLLMClient
    
    class MockVLLMClient:
        async def health_check(self):
            return True
        
        async def chat_completion(self, messages, **kwargs):
            return {
                "content": "Mocked vLLM response",
                "tokens_used": 10,
                "model": "test-model",
                "provider": "vllm",
                "finish_reason": "stop",
                "latency_ms": 50
            }
    
    return MockVLLMClient()


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client"""
    class MockOllamaClient:
        async def health_check(self):
            return True
        
        async def chat_completion(self, messages, **kwargs):
            return {
                "content": "Mocked Ollama response",
                "tokens_used": 15,
                "model": "llama3.2:3b",
                "provider": "ollama",
                "finish_reason": "stop",
                "latency_ms": 200
            }
    
    return MockOllamaClient()


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
    class MockEmbeddingService:
        async def generate(self, texts, **kwargs):
            # Return fake 384-dim embeddings
            embeddings = [[0.1] * 384 for _ in texts]
            return {
                "embeddings": embeddings,
                "model": "test-model",
                "provider": "local",
                "tokens": len(texts) * 5,
                "dimension": 384,
                "processing_time_ms": 50
            }
        
        def get_model_info(self):
            return {
                "name": "test-model",
                "loaded": True,
                "dimension": 384,
                "first_load_time_s": 1.0
            }
    
    return MockEmbeddingService()