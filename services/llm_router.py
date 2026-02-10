"""
LLM Router - Intelligent Provider Routing with Circuit Breaker
EPIC 1.3: Implement LLM Router (vLLM → Ollama → OpenAI)
Routes requests to best available provider with automatic fallback
"""

import logging
import time
from typing import Dict, Any, Optional, List
from enum import Enum

from clients.vllm_client import VLLMClient
from clients.vllm_grpc_client import VLLMGRPCClient
from clients.ollama_client import OllamaClient
from clients.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """LLM provider enumeration"""
    VLLM = "vllm"
    VLLM_GRPC = "vllm_grpc"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Too many failures, skip provider
    HALF_OPEN = "half_open"  # Testing if provider recovered


class AllProvidersFailedError(Exception):
    """Raised when all providers fail"""
    pass


class LLMRouter:
    """
    Intelligent LLM request router with circuit breaker
    
    Routing priority:
    1. vLLM (GPU, local, fast) - FREE
    2. Ollama (CPU, local, slow) - FREE
    3. OpenAI (API, expensive) - PAID
    4. Anthropic (API, expensive) - PAID (future)
    
    Circuit breaker:
    - Opens after 5 consecutive failures
    - Stays open for 5 minutes
    - Auto-recovers after timeout
    """
    
    def __init__(
        self,
        vllm_client: Optional[VLLMClient] = None,
        vllm_grpc_client: Optional[VLLMGRPCClient] = None,
        ollama_client: Optional[OllamaClient] = None,
        openai_client: Optional[OpenAIClient] = None,
        circuit_threshold: int = 5,
        circuit_timeout: int = 300,  # 5 minutes
        health_cache_ttl: int = 30,  # 30 seconds
    ):
        """
        Initialize LLM router
        
        Args:
            vllm_client: vLLM client instance
            ollama_client: Ollama client instance
            openai_client: OpenAI client instance
            circuit_threshold: Failures before opening circuit
            circuit_timeout: Seconds before retrying opened circuit
            health_cache_ttl: Seconds to cache health check results
        """
        self.vllm_client = vllm_client
        self.vllm_grpc_client = vllm_grpc_client
        self.ollama_client = ollama_client
        self.openai_client = openai_client
        
        # Circuit breaker state
        self.circuit_threshold = circuit_threshold
        self.circuit_timeout = circuit_timeout
        self.failures = {p: 0 for p in LLMProvider}
        self.last_failure = {p: None for p in LLMProvider}
        
        # Health check cache
        self.health_cache_ttl = health_cache_ttl
        self.health_cache = {}
        self.last_health_check = {}
        
        logger.info(
            f"✅ LLM Router initialized: "
            f"circuit_threshold={circuit_threshold}, "
            f"circuit_timeout={circuit_timeout}s"
        )
    
    async def infer(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route LLM inference request to best available provider
        
        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            {
                "content": "Generated text",
                "tokens_used": 100,
                "model": "model_name",
                "provider": "vllm",
                "finish_reason": "stop",
                "latency_ms": 150
            }
        
        Raises:
            AllProvidersFailedError: If all providers fail
        """
        # Define provider priority
        providers = [
            (LLMProvider.VLLM_GRPC, self.vllm_grpc_client),
            (LLMProvider.VLLM, self.vllm_client),
            (LLMProvider.OLLAMA, self.ollama_client),
            (LLMProvider.OPENAI, self.openai_client),
        ]
        
        # Track which providers were tried
        tried_providers = []
        errors = {}
        
        for provider, client in providers:
            # Skip if client not configured
            if client is None:
                logger.debug(f"Provider {provider.value} not configured, skipping")
                continue
            
            # Check circuit breaker
            if self._is_circuit_open(provider):
                logger.info(f"Circuit open for {provider.value}, skipping")
                tried_providers.append(f"{provider.value} (circuit open)")
                continue
            
            # Try this provider
            try:
                logger.info(f"Trying provider: {provider.value}")
                start_time = time.time()
                
                result = await client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                result["latency_ms"] = latency_ms
                
                # Success! Reset failure count
                self.failures[provider] = 0
                
                logger.info(
                    f"✅ {provider.value} inference successful: "
                    f"{result['tokens_used']} tokens, {latency_ms}ms"
                )
                
                return result
                
            except TimeoutError as e:
                logger.warning(f"{provider.value} timeout: {e}")
                self._on_failure(provider)
                tried_providers.append(f"{provider.value} (timeout)")
                errors[provider.value] = str(e)
                continue
                
            except Exception as e:
                logger.warning(f"{provider.value} error: {e}")
                self._on_failure(provider)
                tried_providers.append(f"{provider.value} (error)")
                errors[provider.value] = str(e)
                continue
        
        # All providers failed
        logger.error(
            f"❌ All providers failed. Tried: {', '.join(tried_providers)}"
        )
        raise AllProvidersFailedError(
            f"No providers available. Tried: {', '.join(tried_providers)}. "
            f"Errors: {errors}"
        )
    
    def _is_circuit_open(self, provider: LLMProvider) -> bool:
        """
        Check if circuit is open for this provider
        
        Circuit opens after threshold failures and stays open
        for timeout period before auto-recovery attempt
        """
        # If failures below threshold, circuit is closed
        if self.failures[provider] < self.circuit_threshold:
            return False
        
        # Check if timeout has expired (recovery time)
        last_fail = self.last_failure[provider]
        if last_fail is None:
            return False
        
        time_since_failure = time.time() - last_fail
        
        if time_since_failure > self.circuit_timeout:
            # Circuit timeout expired, attempt recovery
            logger.info(
                f"Circuit for {provider.value} timeout expired, attempting recovery"
            )
            self.failures[provider] = self.circuit_threshold - 1  # One more chance
            return False
        
        # Circuit still open
        return True
    
    def _on_failure(self, provider: LLMProvider):
        """Record provider failure"""
        self.failures[provider] += 1
        self.last_failure[provider] = time.time()
        
        if self.failures[provider] >= self.circuit_threshold:
            logger.warning(
                f"⚠️ Circuit opened for {provider.value} "
                f"({self.failures[provider]} failures)"
            )
    
    async def health_check_all(self) -> Dict[str, Any]:
        """
        Check health of all configured providers
        
        Returns:
            {
                "vllm": {"status": "healthy", "latency_ms": 5},
                "ollama": {"status": "healthy", "latency_ms": 100},
                "openai": {"status": "configured"}
            }
        """
        results = {}
        
        # Check vLLM REST
        if self.vllm_client:
            results["vllm"] = await self._check_provider_health(
                LLMProvider.VLLM,
                self.vllm_client
            )
            
        # Check vLLM gRPC
        if self.vllm_grpc_client:
            results["vllm_grpc"] = await self._check_provider_health(
                LLMProvider.VLLM_GRPC,
                self.vllm_grpc_client
            )
        
        # Check Ollama
        if self.ollama_client:
            results["ollama"] = await self._check_provider_health(
                LLMProvider.OLLAMA,
                self.ollama_client
            )
        
        # Check OpenAI
        if self.openai_client:
            results["openai"] = {"status": "configured"}
        
        return results
    
    async def _check_provider_health(
        self,
        provider: LLMProvider,
        client
    ) -> Dict[str, Any]:
        """
        Check health of a specific provider
        
        Uses cache to avoid hammering providers
        """
        # Check cache
        now = time.time()
        if provider in self.last_health_check:
            if (now - self.last_health_check[provider]) < self.health_cache_ttl:
                return self.health_cache.get(provider, {"status": "unknown"})
        
        # Perform health check
        try:
            start_time = time.time()
            is_healthy = await client.health_check()
            latency_ms = int((time.time() - start_time) * 1000)
            
            result = {
                "status": "healthy" if is_healthy else "unhealthy",
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            result = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Update cache
        self.health_cache[provider] = result
        self.last_health_check[provider] = now
        
        return result
    