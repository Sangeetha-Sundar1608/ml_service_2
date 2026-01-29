"""
Tests for LLM Router
EPIC 1.3: LLM Router Testing
"""

import pytest
from services.llm_router import LLMRouter, LLMProvider, AllProvidersFailedError


@pytest.mark.asyncio
async def test_llm_router_vllm_success(mock_vllm_client):
    """Test successful routing to vLLM"""
    router = LLMRouter(vllm_client=mock_vllm_client)
    
    messages = [{"role": "user", "content": "Hello"}]
    result = await router.infer(messages)
    
    assert result["provider"] == "vllm"
    assert result["content"] == "Mocked vLLM response"
    assert result["tokens_used"] == 10


@pytest.mark.asyncio
async def test_llm_router_fallback_to_ollama(mock_ollama_client):
    """Test fallback to Ollama when vLLM unavailable"""
    # vLLM not configured (None)
    router = LLMRouter(vllm_client=None, ollama_client=mock_ollama_client)
    
    messages = [{"role": "user", "content": "Hello"}]
    result = await router.infer(messages)
    
    assert result["provider"] == "ollama"
    assert result["content"] == "Mocked Ollama response"


@pytest.mark.asyncio
async def test_llm_router_all_providers_failed():
    """Test error when all providers fail"""
    router = LLMRouter()  # No clients configured
    
    messages = [{"role": "user", "content": "Hello"}]
    
    with pytest.raises(AllProvidersFailedError):
        await router.infer(messages)


@pytest.mark.asyncio
async def test_circuit_breaker_opens(mock_vllm_client):
    """Test circuit breaker opens after failures"""
    # Mock client that always fails
    class FailingClient:
        async def chat_completion(self, *args, **kwargs):
            raise RuntimeError("Always fails")
    
    router = LLMRouter(
        vllm_client=FailingClient(),
        circuit_threshold=3
    )
    
    messages = [{"role": "user", "content": "Hello"}]
    
    # Trigger failures to open circuit
    for _ in range(3):
        try:
            await router.infer(messages)
        except AllProvidersFailedError:
            pass
    
    # Check circuit is open
    assert router._is_circuit_open(LLMProvider.VLLM) is True


@pytest.mark.asyncio
async def test_health_check_all(mock_vllm_client, mock_ollama_client):
    """Test health check for all providers"""
    router = LLMRouter(
        vllm_client=mock_vllm_client,
        ollama_client=mock_ollama_client
    )
    
    health = await router.health_check_all()
    
    assert "vllm" in health
    assert health["vllm"]["status"] == "healthy"
    assert "ollama" in health
    assert health["ollama"]["status"] == "healthy"