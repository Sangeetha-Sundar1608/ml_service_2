"""
Analytics Routes - Usage & Cost Analytics
EPIC 4.3: Cost Tracking & Analytics
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from datetime import datetime, timedelta
import logging

from models.request_models import TimeRange
from models.response_models import AnalyticsResponse, ProviderUsage
from services.cost_tracker import CostTracker
from dependencies import get_cost_tracker, get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/provider-usage", response_model=AnalyticsResponse)
async def get_provider_usage(
    time_range: TimeRange = Query(TimeRange.DAY, description="Time range"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    db: Session = Depends(get_db),
):
    """
    Get provider usage analytics
    
    **Query Parameters:**
    - `time_range`: hour, day, week, month
    - `provider`: Filter by specific provider (vllm, ollama, openai, anthropic)
    
    **Response:**
```json
    {
      "time_range": "day",
      "providers": [
        {
          "provider": "vllm",
          "requests_total": 1000,
          "requests_success": 995,
          "requests_failed": 5,
          "tokens_total": 50000,
          "latency_avg_ms": 150,
          "latency_p95_ms": 300,
          "cost_usd": 0.0
        },
        {
          "provider": "openai",
          "requests_total": 50,
          "requests_success": 50,
          "requests_failed": 0,
          "tokens_total": 5000,
          "latency_avg_ms": 1200,
          "latency_p95_ms": 2000,
          "cost_usd": 0.15
        }
      ],
      "total_requests": 1050,
      "total_cost_usd": 0.15,
      "error_rate_percent": 0.48
    }
```
    """
    try:
        # Calculate date based on time range
        now = datetime.utcnow()
        
        if time_range == TimeRange.HOUR:
            date = now - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            date = now - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            date = now - timedelta(weeks=1)
        elif time_range == TimeRange.MONTH:
            date = now - timedelta(days=30)
        else:
            date = now - timedelta(days=1)
        
        # Get summary from cost tracker
        summary = await cost_tracker.get_daily_summary(db, date.date())
        
        # Filter by provider if specified
        if provider and provider in summary.get("providers", {}):
            summary["providers"] = {
                provider: summary["providers"][provider]
            }
        
        # Convert to response format
        providers_list = []
        for prov_name, prov_data in summary.get("providers", {}).items():
            providers_list.append(
                ProviderUsage(
                    provider=prov_name,
                    requests_total=prov_data.get("requests", 0),
                    requests_success=prov_data.get("requests", 0),  # TODO: track failures
                    requests_failed=0,
                    tokens_total=prov_data.get("tokens", 0),
                    latency_avg_ms=prov_data.get("avg_latency_ms", 0),
                    latency_p95_ms=prov_data.get("avg_latency_ms", 0) * 1.5,  # Estimate
                    cost_usd=prov_data.get("cost_usd", 0.0),
                )
            )
        
        # Calculate error rate
        total_requests = summary.get("total_requests", 0)
        error_count = summary.get("error_count", 0)
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0.0
        
        return AnalyticsResponse(
            time_range=time_range.value,
            providers=providers_list,
            total_requests=total_requests,
            total_cost_usd=summary.get("total_cost_usd", 0.0),
            error_rate_percent=round(error_rate, 2),
        )
    
    except Exception as e:
        logger.error(f"Analytics query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve analytics",
                "message": str(e),
            }
        )


@router.get("/cost-summary")
async def get_cost_summary(
    days: int = Query(7, ge=1, le=90, description="Number of days"),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    db: Session = Depends(get_db),
):
    """
    Get cost summary for last N days
    
    **Response:**
```json
    {
      "days": 7,
      "total_cost_usd": 12.50,
      "daily_breakdown": [
        {"date": "2025-01-28", "cost_usd": 2.30},
        {"date": "2025-01-27", "cost_usd": 1.80}
      ],
      "by_provider": {
        "openai": 10.00,
        "anthropic": 2.50,
        "vllm": 0.0,
        "ollama": 0.0
      }
    }
```
    """
    try:
        # TODO: Implement multi-day cost summary
        # For now, return today's summary
        today = datetime.utcnow().date()
        summary = await cost_tracker.get_daily_summary(db, today)
        
        return {
            "days": 1,
            "total_cost_usd": summary.get("total_cost_usd", 0.0),
            "daily_breakdown": [
                {
                    "date": summary.get("date"),
                    "cost_usd": summary.get("total_cost_usd", 0.0),
                }
            ],
            "by_provider": {
                prov: data.get("cost_usd", 0.0)
                for prov, data in summary.get("providers", {}).items()
            },
        }
    
    except Exception as e:
        logger.error(f"Cost summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))