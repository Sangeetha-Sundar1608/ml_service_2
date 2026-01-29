"""
Health Routes - Service Health Monitoring
EPIC 4.1: Health Check Endpoint
"""

from fastapi import APIRouter, Depends
import logging

from models.response_models import HealthResponse
from services.health_check import HealthChecker
from main import get_health_checker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health_check(
    health_checker: HealthChecker = Depends(get_health_checker),
):
    """
    Check ML Service health
    
    Returns health status of all components:
    - LLM providers (vLLM, Ollama, OpenAI)
    - Embedding service
    - Document processing
    - System resources
    
    **Response:**
```json
    {
      "status": "healthy",
      "timestamp": "2025-01-28T10:30:00Z",
      "services": {
        "llm_inference": {
          "vllm": {"status": "healthy", "latency_ms": 5},
          "ollama": {"status": "healthy", "latency_ms": 100}
        },
        "embeddings": {"status": "healthy", "model": "all-MiniLM-L6-v2"},
        "document_processing": {"status": "healthy"}
      },
      "resources": {
        "gpu_available": true,
        "gpu_memory_gb": 20.5,
        "cpu_usage_percent": 45.2,
        "memory_usage_gb": 8.1
      },
      "uptime_seconds": 86400
    }
```
    
    **Status Values:**
    - `healthy`: All systems operational
    - `degraded`: Some services unavailable but core working
    - `unhealthy`: Critical services down
    """
    try:
        result = await health_checker.check_all()
        return HealthResponse(**result)
    
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        # Return degraded status if health check itself fails
        return {
            "status": "degraded",
            "timestamp": None,
            "services": {},
            "resources": {},
            "uptime_seconds": 0,
            "error": str(e),
        }