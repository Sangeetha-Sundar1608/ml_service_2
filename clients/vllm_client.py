"""
vLLM Client - REST API (OpenAI-compatible)
Connects to local vLLM GPU server for fast inference
EPIC 1.1: Setup vLLM GPU Server
"""

import httpx
import logging
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for vLLM inference server (OpenAI-compatible REST API)"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "meta-llama/Llama-3.2-3B",
        timeout: float = 120.0,
    ):
        """
        Initialize vLLM client
        
        Args:
            base_url: vLLM server URL (e.g., http://vllm-service:8000)
            model_name: Default model name
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        
        logger.info(f"✅ vLLM client initialized: {base_url} (model: {model_name})")
    
    async def health_check(self) -> bool:
        """
        Check if vLLM server is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # vLLM uses /ping endpoint (without /v1 prefix)
                # base_url is http://vllm:8000/v1, so we need to go up one level
                base_url_without_v1 = self.base_url.rsplit('/v1', 1)[0]
                response = await client.get(f"{base_url_without_v1}/ping")
                is_healthy = response.status_code == 200
                
                if is_healthy:
                    logger.debug("vLLM health check: OK")
                else:
                    logger.warning(f"vLLM health check failed: {response.status_code}")
                
                return is_healthy
                
        except httpx.TimeoutException:
            logger.warning("vLLM health check: timeout")
            return False
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion request (OpenAI format)
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Stream response (not implemented in MVP)
            **kwargs: Additional vLLM parameters
        
        Returns:
            {
                "content": "Generated text",
                "tokens_used": 100,
                "model": "model_name",
                "provider": "vllm",
                "finish_reason": "stop"
            }
        
        Raises:
            TimeoutError: If request times out
            RuntimeError: If vLLM request fails
        """
        try:
            # Prepare OpenAI-compatible request
            payload = {
                "model": kwargs.get("model", self.model_name),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": stream,
            }
            
            # Add optional parameters
            if "stop" in kwargs:
                payload["stop"] = kwargs["stop"]
            
            logger.debug(f"vLLM request: {len(messages)} messages, max_tokens={max_tokens}")
            
            # Make request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Check if base_url already contains /v1 to avoid double prefixing
                url = f"{self.base_url}/chat/completions" if self.base_url.endswith('/v1') else f"{self.base_url}/v1/chat/completions"
                response = await client.post(
                    url,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            # Parse OpenAI-format response
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError("Invalid response format from vLLM")
            
            choice = result["choices"][0]
            content = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason", "stop")
            
            # Extract token usage
            usage = result.get("usage", {})
            tokens_used = usage.get("completion_tokens", 0)
            
            logger.info(
                f"✅ vLLM inference successful: {tokens_used} tokens, "
                f"finish_reason={finish_reason}"
            )
            
            return {
                "content": content,
                "tokens_used": tokens_used,
                "model": result.get("model", self.model_name),
                "provider": "vllm",
                "finish_reason": finish_reason
            }
        
        except httpx.TimeoutException:
            logger.error(f"❌ vLLM request timeout after {self.timeout}s")
            raise TimeoutError(f"vLLM request timeout ({self.timeout}s)")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ vLLM HTTP error: {e.response.status_code}")
            raise RuntimeError(f"vLLM HTTP error: {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"❌ vLLM inference failed: {e}")
            raise RuntimeError(f"vLLM inference failed: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get loaded model information
        
        Returns:
            Model metadata from /v1/models endpoint
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
        