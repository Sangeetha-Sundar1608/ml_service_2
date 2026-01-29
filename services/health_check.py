"""
Health Check Service - Monitor All Providers
EPIC 4.1: Health Check Endpoint
Checks health of all ML service components
"""

import logging
import time
import psutil
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Health monitoring for all ML Service components
    
    Monitors:
    - LLM providers (vLLM, Ollama, OpenAI)
    - Embedding service
    - Document processing
    - System resources (CPU, memory, GPU)
    """
    
    def __init__(
        self,
        vllm_client=None,
        ollama_client=None,
        openai_client=None,
        embedding_service=None,
        document_processor=None,
    ):
        """
        Initialize health checker
        
        Args:
            vllm_client: vLLM client instance
            ollama_client: Ollama client instance
            openai_client: OpenAI client instance
            embedding_service: Embedding service instance
            document_processor: Document processor instance
        """
        self.vllm_client = vllm_client
        self.ollama_client = ollama_client
        self.openai_client = openai_client
        self.embedding_service = embedding_service
        self.document_processor = document_processor
        
        self.start_time = time.time()
        
        logger.info("✅ Health checker initialized")
    
    async def check_all(self) -> Dict[str, Any]:
        """
        Check health of all components
        
        Returns:
            {
                "status": "healthy" | "degraded" | "unhealthy",
                "timestamp": "2025-01-28T10:30:00Z",
                "services": {
                    "llm_inference": {...},
                    "embeddings": {...},
                    "document_processing": {...}
                },
                "resources": {...},
                "uptime_seconds": 86400
            }
        """
        start_time = time.time()
        
        # Check all services
        services = {
            "llm_inference": await self._check_llm_inference(),
            "embeddings": await self._check_embeddings(),
            "document_processing": await self._check_document_processing(),
        }
        
        # Check system resources
        resources = self._check_resources()
        
        # Determine overall status
        overall_status = self._determine_overall_status(services)
        
        check_time_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": services,
            "resources": resources,
            "uptime_seconds": int(time.time() - self.start_time),
            "health_check_time_ms": check_time_ms,
        }
        
        logger.debug(f"Health check complete: {overall_status} ({check_time_ms}ms)")
        
        return result
    
    async def _check_llm_inference(self) -> Dict[str, Any]:
        """
        Check LLM inference providers
        
        Returns:
            {
                "vllm": {"status": "healthy", "latency_ms": 5},
                "ollama": {"status": "healthy", "latency_ms": 100},
                "openai": {"status": "configured"}
            }
        """
        providers = {}
        
        # Check vLLM
        if self.vllm_client:
            providers["vllm"] = await self._check_provider(
                "vLLM",
                self.vllm_client.health_check
            )
        
        # Check Ollama
        if self.ollama_client:
            providers["ollama"] = await self._check_provider(
                "Ollama",
                self.ollama_client.health_check
            )
        
        # Check OpenAI (just check if configured)
        if self.openai_client:
            providers["openai"] = {"status": "configured"}
        
        return providers
    
    async def _check_embeddings(self) -> Dict[str, Any]:
        """
        Check embedding service
        
        Returns:
            {
                "status": "healthy",
                "model": "all-MiniLM-L6-v2",
                "model_loaded": True,
                "dimension": 384
            }
        """
        if not self.embedding_service:
            return {"status": "not_configured"}
        
        try:
            # Get model info (doesn't actually load if not loaded yet)
            model_info = self.embedding_service.get_model_info()
            
            return {
                "status": "healthy",
                "model": model_info["name"],
                "model_loaded": model_info["loaded"],
                "dimension": model_info["dimension"],
            }
        
        except Exception as e:
            logger.warning(f"Embedding service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_document_processing(self) -> Dict[str, Any]:
        """
        Check document processing service
        
        Returns:
            {
                "status": "healthy",
                "processors": ["pdf", "docx", "ocr"]
            }
        """
        if not self.document_processor:
            return {"status": "not_configured"}
        
        try:
            # Check if processors are initialized
            processors = []
            
            if hasattr(self.document_processor, 'pdf_processor'):
                processors.append("pdf")
            if hasattr(self.document_processor, 'docx_processor'):
                processors.append("docx")
            if hasattr(self.document_processor, 'ocr_processor'):
                processors.append("ocr")
            
            return {
                "status": "healthy",
                "processors": processors
            }
        
        except Exception as e:
            logger.warning(f"Document processing health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_provider(
        self,
        name: str,
        health_check_fn
    ) -> Dict[str, Any]:
        """
        Check health of a single provider
        
        Args:
            name: Provider name (for logging)
            health_check_fn: Async function to call
        
        Returns:
            {"status": "healthy", "latency_ms": 10}
            or
            {"status": "unhealthy", "error": "..."}
        """
        try:
            start_time = time.time()
            is_healthy = await health_check_fn()
            latency_ms = int((time.time() - start_time) * 1000)
            
            if is_healthy:
                return {
                    "status": "healthy",
                    "latency_ms": latency_ms
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Health check returned False"
                }
        
        except Exception as e:
            logger.warning(f"{name} health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_resources(self) -> Dict[str, Any]:
        """
        Check system resources
        
        Returns:
            {
                "gpu_available": True,
                "gpu_memory_gb": 20.5,
                "gpu_memory_total_gb": 24.0,
                "cpu_usage_percent": 45.2,
                "memory_usage_gb": 8.1,
                "memory_total_gb": 32.0
            }
        """
        resources = {}
        
        # CPU usage
        resources["cpu_usage_percent"] = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        resources["memory_usage_gb"] = round(memory.used / (1024**3), 2)
        resources["memory_total_gb"] = round(memory.total / (1024**3), 2)
        resources["memory_usage_percent"] = memory.percent
        
        # GPU info (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu = gpus[0]  # First GPU
                resources["gpu_available"] = True
                resources["gpu_memory_gb"] = round(gpu.memoryUsed / 1024, 2)
                resources["gpu_memory_total_gb"] = round(gpu.memoryTotal / 1024, 2)
                resources["gpu_memory_percent"] = round(
                    (gpu.memoryUsed / gpu.memoryTotal) * 100, 1
                )
                resources["gpu_name"] = gpu.name
            else:
                resources["gpu_available"] = False
        
        except ImportError:
            # GPUtil not installed
            resources["gpu_available"] = False
        except Exception as e:
            logger.debug(f"GPU check failed: {e}")
            resources["gpu_available"] = False
        
        return resources
    
    def _determine_overall_status(
        self,
        services: Dict[str, Any]
    ) -> str:
        """
        Determine overall health status
        
        Rules:
        - healthy: All critical services OK
        - degraded: Some services down but core working
        - unhealthy: Critical services down
        
        Args:
            services: Service health check results
        
        Returns:
            "healthy" | "degraded" | "unhealthy"
        """
        # Count unhealthy services
        unhealthy_count = 0
        total_count = 0
        
        # Check LLM providers (at least one should be healthy)
        llm = services.get("llm_inference", {})
        llm_healthy = False
        
        for provider, status in llm.items():
            total_count += 1
            if isinstance(status, dict):
                if status.get("status") == "healthy" or status.get("status") == "configured":
                    llm_healthy = True
                else:
                    unhealthy_count += 1
        
        # Check embeddings (should be healthy)
        embeddings = services.get("embeddings", {})
        if embeddings.get("status") != "healthy":
            unhealthy_count += 1
        total_count += 1
        
        # Determine status
        if not llm_healthy:
            # No LLM providers available - critical
            return "unhealthy"
        
        if unhealthy_count == 0:
            return "healthy"
        
        if unhealthy_count < total_count / 2:
            return "degraded"
        
        return "unhealthy"
    