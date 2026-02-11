"""
Cost Tracker - Provider Usage & Cost Analytics
EPIC 4.3: Cost Tracking & Analytics
Tracks all provider usage and calculates costs
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from models.db_models import ProviderUsageLog, DailyCostSummary, ProviderEnum

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Track provider usage and calculate costs
    
    Responsibilities:
    - Log each provider request
    - Calculate API costs (OpenAI, Anthropic)
    - Track daily cost summaries
    - Alert on cost thresholds
    """
    
    def __init__(
        self,
        openai_daily_alert: float = 50.0,
        embedding_daily_alert: float = 10.0,
    ):
        """
        Initialize cost tracker
        
        Args:
            openai_daily_alert: Alert if OpenAI cost > this USD/day
            embedding_daily_alert: Alert if embedding cost > this USD/day
        """
        self.openai_daily_alert = openai_daily_alert
        self.embedding_daily_alert = embedding_daily_alert
        
        logger.info(
            f"✅ Cost tracker initialized: "
            f"OpenAI alert=${openai_daily_alert}/day, "
            f"Embedding alert=${embedding_daily_alert}/day"
        )
    
    async def track_llm_usage(
        self,
        db: Session,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        success: bool = True,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Track LLM inference usage
        
        Args:
            db: Database session
            provider: Provider name (vllm, ollama, openai, anthropic)
            model: Model name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            latency_ms: Request latency
            success: Whether request succeeded
            error_message: Error message if failed
            request_id: Optional request ID for tracing
        """
        try:
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost (only for API providers)
            cost_usd = self._calculate_llm_cost(
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # Log to database
            usage_log = ProviderUsageLog(
                provider=ProviderEnum(provider),
                model=model,
                request_type="llm",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                success=success,
                cost_usd=cost_usd,
                error_message=error_message,
                request_id=request_id
            )
            
            db.add(usage_log)
            db.commit()
            
            # Check daily cost alerts
            if provider in ["openai", "anthropic"]:
                await self._check_daily_cost_alert(db, provider, cost_usd)
            
            logger.debug(
                f"Tracked {provider} usage: {total_tokens} tokens, "
                f"cost=${cost_usd:.6f}, latency={latency_ms}ms"
            )
            
        except Exception as e:
            logger.error(f"Failed to track usage: {e}")
            db.rollback()
    
    async def track_embedding_usage(
        self,
        db: Session,
        provider: str,
        model: str,
        tokens: int,
        latency_ms: int,
        success: bool = True,
    ) -> None:
        """
        Track embedding generation usage
        
        Args:
            db: Database session
            provider: Provider (local or openai)
            model: Model name
            tokens: Total tokens processed
            latency_ms: Request latency
            success: Whether request succeeded
        """
        try:
            # Calculate cost
            cost_usd = self._calculate_embedding_cost(provider, model, tokens)
            
            # Log to database
            usage_log = ProviderUsageLog(
                provider=ProviderEnum.OPENAI if provider == "openai" else ProviderEnum.VLLM,  # Use VLLM for local
                model=model,
                request_type="embedding",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=tokens,
                latency_ms=latency_ms,
                success=success,
                cost_usd=cost_usd
            )
            
            db.add(usage_log)
            db.commit()
            
            # Check alerts for OpenAI embeddings
            if provider == "openai":
                await self._check_embedding_cost_alert(db, cost_usd)
            
            logger.debug(
                f"Tracked {provider} embedding: {tokens} tokens, cost=${cost_usd:.6f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to track embedding usage: {e}")
            db.rollback()
    
    def _calculate_llm_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate LLM inference cost in USD
        
        Pricing (Jan 2025):
        OpenAI:
          - gpt-4o-mini: $0.150/1M input, $0.600/1M output
          - gpt-4o: $2.50/1M input, $10.00/1M output
          - gpt-3.5-turbo: $0.50/1M input, $1.50/1M output
        
        Anthropic:
          - claude-3-5-sonnet: $3.00/1M input, $15.00/1M output
          - claude-3-haiku: $0.25/1M input, $1.25/1M output
        
        Local (vLLM, Ollama): $0.00
        """
        if provider in ["vllm", "ollama"]:
            return 0.0  # Free (local)
        
        if provider == "openai":
            pricing = {
                "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
                "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
                "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
            }
            rates = pricing.get(model, pricing["gpt-4o-mini"])  # Default to mini
            
        elif provider == "anthropic":
            pricing = {
                "claude-3-5-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
                "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
            }
            rates = pricing.get(model, pricing["claude-3-haiku"])  # Default to haiku
            
        else:
            return 0.0
        
        cost = (prompt_tokens * rates["input"]) + (completion_tokens * rates["output"])
        return round(cost, 8)
    
    def _calculate_embedding_cost(
        self,
        provider: str,
        model: str,
        tokens: int
    ) -> float:
        """
        Calculate embedding cost
        
        Pricing:
        OpenAI:
          - text-embedding-3-small: $0.020/1M tokens
          - text-embedding-3-large: $0.130/1M tokens
        
        Local: $0.00
        """
        if provider != "openai":
            return 0.0
        
        pricing = {
            "text-embedding-3-small": 0.020 / 1_000_000,
            "text-embedding-3-large": 0.130 / 1_000_000,
        }
        
        rate = pricing.get(model, pricing["text-embedding-3-small"])
        return round(tokens * rate, 8)
    
    async def _check_daily_cost_alert(
        self,
        db: Session,
        provider: str,
        new_cost: float
    ) -> None:
        """Check if daily cost exceeds alert threshold"""
        try:
            # Get today's total cost for this provider
            today = datetime.utcnow().date()
            
            daily_cost = db.query(
                func.sum(ProviderUsageLog.cost_usd)
            ).filter(
                ProviderUsageLog.provider == ProviderEnum(provider),
                func.date(ProviderUsageLog.created_at) == today
            ).scalar() or 0.0
            
            # Check threshold
            if provider == "openai" and daily_cost > self.openai_daily_alert:
                logger.warning(
                    f"⚠️ OpenAI daily cost alert: ${daily_cost:.2f} "
                    f"(threshold: ${self.openai_daily_alert})"
                )
            
        except Exception as e:
            logger.error(f"Failed to check cost alert: {e}")
    
    async def _check_embedding_cost_alert(
        self,
        db: Session,
        new_cost: float
    ) -> None:
        """Check if daily embedding cost exceeds threshold"""
        try:
            today = datetime.utcnow().date()
            
            daily_cost = db.query(
                func.sum(ProviderUsageLog.cost_usd)
            ).filter(
                ProviderUsageLog.request_type == "embedding",
                func.date(ProviderUsageLog.created_at) == today
            ).scalar() or 0.0
            
            if daily_cost > self.embedding_daily_alert:
                logger.warning(
                    f"⚠️ Embedding daily cost alert: ${daily_cost:.2f} "
                    f"(threshold: ${self.embedding_daily_alert})"
                )
            
        except Exception as e:
            logger.error(f"Failed to check embedding cost alert: {e}")
    
    async def get_daily_summary(
        self,
        db: Session,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get cost summary for a specific date
        
        Args:
            db: Database session
            date: Date to query (default: today)
        
        Returns:
            Dictionary with usage and cost breakdown
        """
        if date is None:
            date = datetime.utcnow().date()
        
        try:
            # Query usage logs for this date
            logs = db.query(ProviderUsageLog).filter(
                func.date(ProviderUsageLog.created_at) == date
            ).all()
            
            # Aggregate by provider
            summary = {
                "date": date.isoformat(),
                "providers": {},
                "total_requests": len(logs),
                "total_tokens": sum(log.total_tokens for log in logs),
                "total_cost_usd": sum(log.cost_usd for log in logs),
                "error_count": sum(1 for log in logs if not log.success),
            }
            
            # Per-provider breakdown
            for provider_enum in ProviderEnum:
                provider = provider_enum.value
                provider_logs = [log for log in logs if log.provider == provider_enum]
                
                if provider_logs:
                    # latency calc
                    latencies = sorted([log.latency_ms for log in provider_logs])
                    avg_latency = sum(latencies) / len(latencies)
                    p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0
                    
                    success_count = sum(1 for log in provider_logs if log.success)
                    error_count = len(provider_logs) - success_count
                    
                    summary["providers"][provider] = {
                        "requests": len(provider_logs),
                        "requests_success": success_count,
                        "requests_failed": error_count,
                        "tokens": sum(log.total_tokens for log in provider_logs),
                        "cost_usd": sum(log.cost_usd for log in provider_logs),
                        "avg_latency_ms": avg_latency,
                        "p95_latency_ms": p95_latency,
                    }
                    
                    # Check performance alerts immediately for this batch
                    self._check_performance_alerts(provider, error_count, len(provider_logs), p95_latency)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get daily summary: {e}")
            return {}

    def _check_performance_alerts(
        self,
        provider: str,
        error_count: int,
        total_count: int,
        p95_latency: float
    ) -> None:
        """Check for error rate and latency alerts"""
        # Alert if error rate > 5% (min 20 requests to avoid noise)
        if total_count >= 20:
            error_rate = (error_count / total_count) * 100
            if error_rate > 5.0:
                logger.warning(
                    f"⚠️ High error rate for {provider}: {error_rate:.1f}% "
                    f"({error_count}/{total_count} failed)"
                )
        
        # Alert if p95 latency > 2 seconds (2000ms)
        if p95_latency > 2000:
            logger.warning(
                f"⚠️ High latency for {provider}: p95={p95_latency}ms "
                f"(threshold: 2000ms)"
            )
        