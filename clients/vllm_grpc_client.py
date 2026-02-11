"""
vLLM gRPC Client - High-performance Inference
Connects to vLLM gRPC server (port 9000)
"""

import grpc
import logging
import sys
import os
from typing import Dict, Any, List, Optional
import time

# Add generated directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'generated'))

try:
    import vllm_pb2
    import vllm_pb2_grpc
except ImportError:
    vllm_pb2 = None
    vllm_pb2_grpc = None
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ vLLM gRPC stubs not found. Using mocks or waiting for build.")

logger = logging.getLogger(__name__)


class VLLMGRPCClient:
    """Client for vLLM gRPC inference server"""
    
    def __init__(
        self,
        address: str = "vllm:9000",
        model_name: str = "meta-llama/Llama-3.2-3B",
        timeout: float = 120.0,
    ):
        """
        Initialize vLLM gRPC client
        
        Args:
            address: vLLM gRPC server address (e.g., vllm:9000)
            model_name: Default model name
            timeout: Request timeout in seconds
        """
        if vllm_pb2_grpc is None:
            logger.error("❌ vLLM gRPC stubs not found! Cannot initialize gRPC client.")
            raise ImportError("vLLM gRPC stubs not found. Run the build process to generate them.")

        self.address = address
        self.model_name = model_name
        self.timeout = timeout
        self.channel = grpc.aio.insecure_channel(address)
        self.stub = vllm_pb2_grpc.InferenceServiceStub(self.channel)
        
        logger.info(f"✅ vLLM gRPC client initialized: {address} (model: {model_name})")
    
    async def health_check(self) -> bool:
        """
        Check if vLLM gRPC server is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            request = vllm_pb2.HealthRequest()
            response = await self.stub.Health(request, timeout=5.0)
            return response.healthy
        except Exception as e:
            logger.warning(f"vLLM gRPC health check failed: {e}")
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform chat completion via gRPC
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            Standard inference response dict
        """
        try:
            # Prepare gRPC request
            grpc_messages = [
                vllm_pb2.Message(role=m["role"], content=m["content"])
                for m in messages
            ]
            
            request = vllm_pb2.InferRequest(
                model=kwargs.get("model", self.model_name),
                messages=grpc_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            # Make request
            start_time = time.time()
            response = await self.stub.Infer(request, timeout=self.timeout)
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "content": response.content,
                "tokens_used": response.tokens_used,
                "model": response.model,
                "provider": "vllm_grpc",
                "finish_reason": response.finish_reason,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f"❌ vLLM gRPC inference failed: {e}")
            raise RuntimeError(f"vLLM gRPC inference failed: {str(e)}")
    
    async def close(self):
        """Close gRPC channel"""
        await self.channel.close()
