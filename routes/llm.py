"""
LLM Routes - LLM Inference Endpoints
EPIC 1: LLM Inference Foundation
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from models.request_models import LLMInferenceRequest
from models.response_models import LLMInferenceResponse
from services.llm_router import LLMRouter, AllProvidersFailedError
from services.cost_tracker import CostTracker
from dependencies import get_llm_router, get_cost_tracker, get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer", response_model=LLMInferenceResponse)
async def llm_inference(
    request: LLMInferenceRequest,
    llm_router: LLMRouter = Depends(get_llm_router),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    db: Session = Depends(get_db),
):
    """
    LLM inference with automatic provider routing
    
    Routes through providers in priority order:
    1. vLLM (GPU, local, fast) - FREE
    2. Ollama (CPU, local, slow) - FREE
    3. OpenAI (API, expensive) - PAID
    
    **Request:**
```json
    {
      "messages": [
        {"role": "user", "content": "Hello, how are you?"}
      ],
      "temperature": 0.7,
      "max_tokens": 256
    }
```
    
    **Response:**
```json
    {
      "content": "I'm doing well, thank you!",
      "tokens_used": 15,
      "model": "meta-llama/Llama-3.2-3B",
      "provider": "vllm",
      "finish_reason": "stop",
      "latency_ms": 150
    }
```
    """
    try:
        # Convert Pydantic messages to dicts
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        logger.info(f"LLM inference request: {len(messages)} messages")
        
        # Route request through providers
        result = await llm_router.infer(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )
        
        # Track usage in database
        await cost_tracker.track_llm_usage(
            db=db,
            provider=result["provider"],
            model=result["model"],
            prompt_tokens=result.get("tokens_used", 0) // 2,  # Estimate
            completion_tokens=result.get("tokens_used", 0) // 2,
            latency_ms=result.get("latency_ms", 0),
            success=True,
        )
        
        logger.info(
            f"✅ LLM inference complete: provider={result['provider']}, "
            f"tokens={result['tokens_used']}"
        )
        
        return LLMInferenceResponse(**result)
    
    except AllProvidersFailedError as e:
        logger.error(f"All providers failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "All LLM providers unavailable",
                "message": str(e),
            }
        )
    
    except Exception as e:
        logger.error(f"LLM inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
            }
        )


@router.get("/providers")
async def get_provider_status(
    llm_router: LLMRouter = Depends(get_llm_router),
):
    """
    Get status of all LLM providers
    
    **Response:**
```json
    {
      "vllm": {"status": "healthy", "latency_ms": 5},
      "ollama": {"status": "healthy", "latency_ms": 100},
      "openai": {"status": "configured"}
    }
```
    """
    try:
        status = await llm_router.health_check_all()
        return status
    
    except Exception as e:
        logger.error(f"Provider status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))