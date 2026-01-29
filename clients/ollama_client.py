"""
Ollama Client - CPU Fallback
CPU-based LLM inference using Ollama
EPIC 1.2: Setup Ollama CPU Fallback
"""

import httpx
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama inference (CPU fallback)"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama server URL
            model_name: Model name (e.g., llama3.2:3b, llama2:7b)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        
        logger.info(f"✅ Ollama client initialized: {base_url} (model: {model_name})")
    
    async def health_check(self) -> bool:
        """
        Check if Ollama server is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Ollama doesn't have /health, so we use /api/tags (list models)
                response = await client.get(f"{self.base_url}/api/tags")
                is_healthy = response.status_code == 200
                
                if is_healthy:
                    logger.debug("Ollama health check: OK")
                else:
                    logger.warning(f"Ollama health check failed: {response.status_code}")
                
                return is_healthy
                
        except httpx.TimeoutException:
            logger.warning("Ollama health check: timeout")
            return False
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion using Ollama
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            {
                "content": "Generated text",
                "tokens_used": 100,
                "model": "model_name",
                "provider": "ollama",
                "finish_reason": "stop"
            }
        
        Raises:
            TimeoutError: If request times out
            RuntimeError: If Ollama request fails
        """
        try:
            # Convert messages to Ollama format (single prompt)
            prompt = self._format_messages_to_prompt(messages)
            
            # Prepare Ollama request
            payload = {
                "model": kwargs.get("model", self.model_name),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens
                }
            }
            
            logger.debug(f"Ollama request: prompt_len={len(prompt)}, max_tokens={max_tokens}")
            
            # Make request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            # Parse Ollama response
            content = result.get("response", "")
            
            # Estimate tokens (Ollama doesn't always return exact count)
            tokens_used = result.get("eval_count", len(content.split()))
            
            logger.info(
                f"✅ Ollama inference successful: ~{tokens_used} tokens"
            )
            
            return {
                "content": content,
                "tokens_used": tokens_used,
                "model": result.get("model", self.model_name),
                "provider": "ollama",
                "finish_reason": "stop" if result.get("done", False) else "length"
            }
        
        except httpx.TimeoutException:
            logger.error(f"❌ Ollama request timeout after {self.timeout}s")
            raise TimeoutError(f"Ollama request timeout ({self.timeout}s)")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Ollama HTTP error: {e.response.status_code}")
            raise RuntimeError(f"Ollama HTTP error: {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"❌ Ollama inference failed: {e}")
            raise RuntimeError(f"Ollama inference failed: {str(e)}")
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to single prompt
        
        Args:
            messages: [{"role": "user", "content": "..."}]
        
        Returns:
            Single formatted prompt string
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    async def list_models(self) -> List[str]:
        """
        List available Ollama models
        
        Returns:
            List of model names
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                result = response.json()
                
                models = [model["name"] for model in result.get("models", [])]
                return models
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []