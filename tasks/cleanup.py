"""
Celery Tasks - Cleanup & Maintenance
Periodic cleanup of old jobs, temp files, cached data
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from celery import Task

from celery_app import celery_app
from models.db_models import AsyncJob, ProviderUsageLog
from dependencies import SessionLocal

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.cleanup_old_jobs")
def cleanup_old_jobs(days: int = 7):
    """
    Clean up old async jobs from database
    
    Removes jobs older than specified days
    
    Args:
        days: Keep jobs from last N days
    """
    logger.info(f"Starting cleanup of jobs older than {days} days")
    
    db = SessionLocal()
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Delete old completed/failed jobs
        deleted = db.query(AsyncJob).filter(
            AsyncJob.created_at < cutoff_date,
            AsyncJob.status.in_(["completed", "failed"])
        ).delete()
        
        db.commit()
        
        logger.info(f"✅ Cleaned up {deleted} old jobs")
        
        return {
            "deleted_jobs": deleted,
            "cutoff_date": cutoff_date.isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        db.rollback()
        raise
    
    finally:
        db.close()


@celery_app.task(name="tasks.cleanup_temp_files")
def cleanup_temp_files(temp_dir: str = "/tmp", hours: int = 24):
    """
    Clean up old temporary files
    
    Removes temp files older than specified hours
    
    Args:
        temp_dir: Temporary directory to clean
        hours: Remove files older than N hours
    """
    logger.info(f"Starting cleanup of temp files in {temp_dir}")
    
    try:
        temp_path = Path(temp_dir)
        cutoff_time = time.time() - (hours * 3600)
        
        deleted_count = 0
        deleted_size = 0
        
        # Find and delete old temp files
        for file_path in temp_path.glob("ml_service_*"):
            if file_path.is_file():
                try:
                    # Check file age
                    file_mtime = file_path.stat().st_mtime
                    
                    if file_mtime < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        deleted_size += file_size
                        
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
                    continue
        
        logger.info(
            f"✅ Cleaned up {deleted_count} temp files "
            f"({deleted_size / 1024 / 1024:.2f} MB)"
        )
        
        return {
            "deleted_files": deleted_count,
            "deleted_size_mb": round(deleted_size / 1024 / 1024, 2),
        }
        
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}", exc_info=True)
        raise


@celery_app.task(name="tasks.aggregate_daily_costs")
def aggregate_daily_costs():
    """
    Aggregate daily cost summaries from usage logs
    
    Creates/updates DailyCostSummary records
    """
    logger.info("Starting daily cost aggregation")
    
    db = SessionLocal()
    
    try:
        from sqlalchemy import func
        from models.db_models import DailyCostSummary, ProviderEnum
        
        today = datetime.utcnow().date()
        
        # Aggregate today's usage
        usage_stats = db.query(
            ProviderUsageLog.provider,
            func.count(ProviderUsageLog.id).label('requests'),
            func.sum(ProviderUsageLog.total_tokens).label('tokens'),
            func.sum(ProviderUsageLog.cost_usd).label('cost'),
            func.avg(ProviderUsageLog.latency_ms).label('avg_latency'),
            func.count(
                func.nullif(ProviderUsageLog.success, True)
            ).label('errors'),
        ).filter(
            func.date(ProviderUsageLog.created_at) == today
        ).group_by(
            ProviderUsageLog.provider
        ).all()
        
        # Check if summary exists
        summary = db.query(DailyCostSummary).filter(
            func.date(DailyCostSummary.date) == today
        ).first()
        
        if not summary:
            summary = DailyCostSummary(date=datetime.combine(today, datetime.min.time()))
            db.add(summary)
        
        # Update summary
        total_requests = 0
        total_tokens = 0
        total_cost = 0.0
        error_count = 0
        
        for stat in usage_stats:
            provider = stat.provider.value
            
            # Update provider-specific fields
            setattr(summary, f"{provider}_requests", stat.requests or 0)
            setattr(summary, f"{provider}_tokens", stat.tokens or 0)
            
            if provider in ["openai", "anthropic"]:
                setattr(summary, f"{provider}_cost_usd", stat.cost or 0.0)
            
            total_requests += stat.requests or 0
            total_tokens += stat.tokens or 0
            total_cost += stat.cost or 0.0
            error_count += stat.errors or 0
        
        # Update totals
        summary.total_requests = total_requests
        summary.total_tokens = total_tokens
        summary.total_cost_usd = total_cost
        summary.error_count = error_count
        
        db.commit()
        
        logger.info(
            f"✅ Daily cost summary updated: "
            f"{total_requests} requests, ${total_cost:.4f}"
        )
        
        return {
            "date": today.isoformat(),
            "total_requests": total_requests,
            "total_cost_usd": total_cost,
        }
        
    except Exception as e:
        logger.error(f"Cost aggregation failed: {e}", exc_info=True)
        db.rollback()
        raise
    
    finally:
        db.close()


# Schedule periodic tasks
celery_app.conf.beat_schedule = {
    'cleanup-old-jobs-daily': {
        'task': 'tasks.cleanup_old_jobs',
        'schedule': 86400.0,  # Run daily (24 hours)
        'args': (7,),  # Keep last 7 days
    },
    'cleanup-temp-files-hourly': {
        'task': 'tasks.cleanup_temp_files',
        'schedule': 3600.0,  # Run hourly
        'args': ('/tmp', 24),  # Remove files older than 24 hours
    },
    'aggregate-costs-hourly': {
        'task': 'tasks.aggregate_daily_costs',
        'schedule': 3600.0,  # Run hourly
    },
}