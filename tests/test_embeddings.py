"""
Tests for Embedding Service
EPIC 3: Embedding Testing
"""

import pytest


@pytest.mark.asyncio
async def test_embedding_generation(mock_embedding_service):
    """Test local embedding generation"""
    texts = ["Hello world", "Test text"]
    result = await mock_embedding_service.generate(texts)
    
    assert len(result["embeddings"]) == 2
    assert len(result["embeddings"][0]) == 384
    assert result["provider"] == "local"
    assert result["dimension"] == 384


@pytest.mark.asyncio
async def test_empty_texts_error(mock_embedding_service):
    """Test error on empty texts"""
    with pytest.raises(ValueError):
        await mock_embedding_service.generate([])


def test_model_info(mock_embedding_service):
    """Test model info retrieval"""
    info = mock_embedding_service.get_model_info()
    
    assert info["name"] == "test-model"
    assert info["loaded"] is True
    assert info["dimension"] == 384