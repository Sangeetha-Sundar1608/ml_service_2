"""
OpenAI Client - API Fallback
Expensive but reliable fallback provider
EPIC 1.3: LLM Router - OpenAI fallback
"""

import httpx
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for OpenAI API (expensive fallback)"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            model_name: Model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        self.model_name = model_name
        self.timeout = timeout
        
        logger.info(f"✅ OpenAI client initialized (model: {model_name})")
    
    async def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible
        
        Returns:
            True if API key is valid and accessible
        """
        try:
            # Simple test: list models
            await self.client.models.list()
            logger.debug("OpenAI health check: OK")
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion using OpenAI API
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
        
        Returns:
            {
                "content": "Generated text",
                "tokens_used": 100,
                "model": "model_name",
                "provider": "openai",
                "finish_reason": "stop",
                "cost_usd": 0.0001
            }
        
        Raises:
            TimeoutError: If request times out
            RuntimeError: If OpenAI request fails
        """
        try:
            logger.debug(f"OpenAI request: {len(messages)} messages, max_tokens={max_tokens}")
            
            # Make request using official OpenAI SDK
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.model_name),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract response
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            
            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            # Calculate cost (approximate)
            cost_usd = self._calculate_cost(
                model=response.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            logger.info(
                f"✅ OpenAI inference successful: {total_tokens} tokens, "
                f"cost=${cost_usd:.6f}"
            )
            
            return {
                "content": content,
                "tokens_used": total_tokens,
                "model": response.model,
                "provider": "openai",
                "finish_reason": finish_reason,
                "cost_usd": cost_usd
            }
        
        except Exception as e:
            logger.error(f"❌ OpenAI inference failed: {e}")
            raise RuntimeError(f"OpenAI inference failed: {str(e)}")
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """
        Generate embeddings using OpenAI
        EPIC 3.3: OpenAI Embedding Fallback
        
        Args:
            texts: List of texts to embed
            model: Embedding model (text-embedding-3-small or text-embedding-3-large)
        
        Returns:
            List of embedding vectors
        """
        try:
            logger.debug(f"OpenAI embeddings: {len(texts)} texts")
            
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            # Calculate cost
            total_tokens = response.usage.total_tokens
            cost_usd = self._calculate_embedding_cost(model, total_tokens)
            
            logger.info(
                f"✅ OpenAI embeddings successful: {len(embeddings)} vectors, "
                f"cost=${cost_usd:.6f}"
            )
            
            return embeddings
        
        except Exception as e:
            logger.error(f"❌ OpenAI embeddings failed: {e}")
            raise RuntimeError(f"OpenAI embeddings failed: {str(e)}")
    
    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate cost in USD
        
        Pricing (as of Jan 2025):
        - gpt-4o-mini: $0.150/1M input, $0.600/1M output
        - gpt-4o: $2.50/1M input, $10.00/1M output
        - gpt-3.5-turbo: $0.50/1M input, $1.50/1M output
        """
        pricing = {
            "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
        }
        
        # Default to gpt-4o-mini pricing
        rates = pricing.get(model, pricing["gpt-4o-mini"])
        
        cost = (prompt_tokens * rates["input"]) + (completion_tokens * rates["output"])
        return cost
    
    def _calculate_embedding_cost(self, model: str, tokens: int) -> float:
        """
        Calculate embedding cost
        
        Pricing:
        - text-embedding-3-small: $0.020/1M tokens
        - text-embedding-3-large: $0.130/1M tokens
        """
        pricing = {
            "text-embedding-3-small": 0.020 / 1_000_000,
            "text-embedding-3-large": 0.130 / 1_000_000,
        }
        
        rate = pricing.get(model, pricing["text-embedding-3-small"])
        return tokens * rate