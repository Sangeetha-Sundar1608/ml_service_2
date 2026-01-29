"""
Celery Tasks - Batch Embedding Processing
EPIC 3.2: Batch Embedding with Optimization
Async processing of large embedding batches
"""

import logging
import time
import json
from typing import List, Dict, Any
from celery import Task

from celery_app import celery_app
from services.embedding_service import EmbeddingService
from clients.openai_client import OpenAIClient
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingTask(Task):
    """Base task with shared embedding service"""
    _embedding_service = None
    
    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            # Initialize OpenAI client if configured
            openai_client = None
            if settings.OPENAI_API_KEY:
                openai_client = OpenAIClient(
                    api_key=settings.OPENAI_API_KEY,
                    model_name="gpt-4o-mini"
                )
            
            self._embedding_service = EmbeddingService(
                model_name=settings.EMBEDDING_MODEL,
                openai_client=openai_client,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
            )
        return self._embedding_service


@celery_app.task(
    bind=True,
    base=EmbeddingTask,
    name="tasks.process_batch_embeddings",
    max_retries=3,
    default_retry_delay=60,
)
def process_batch_embeddings(
    self,
    job_id: str,
    texts: List[str],
    chunk_size: int = 1000,
) -> Dict[str, Any]:
    """
    Process batch embeddings asynchronously
    
    Args:
        job_id: Unique job identifier
        texts: List of texts to embed (up to 10,000)
        chunk_size: Process in chunks of this size
    
    Returns:
        {
            "job_id": "uuid",
            "status": "completed",
            "total_texts": 10000,
            "embeddings_count": 10000,
            "processing_time_s": 30.5,
            "chunks_processed": 10,
            "failed_chunks": 0
        }
    """
    logger.info(f"[{job_id}] Starting batch embedding: {len(texts)} texts")
    
    start_time = time.time()
    
    try:
        # Process in chunks
        all_embeddings = []
        chunks_processed = 0
        failed_chunks = []
        
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            logger.info(
                f"[{job_id}] Processing chunk {chunk_num}/{total_chunks} "
                f"({len(chunk)} texts)"
            )
            
            try:
                # Generate embeddings for chunk
                # Use asyncio.run for async function in sync Celery task
                import asyncio
                result = asyncio.run(
                    self.embedding_service.generate(
                        texts=chunk,
                        normalize=True,
                        use_fallback=True,
                    )
                )
                
                all_embeddings.extend(result["embeddings"])
                chunks_processed += 1
                
                # Update progress
                progress = int((chunks_processed / total_chunks) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': chunks_processed,
                        'total': total_chunks,
                        'progress': progress,
                    }
                )
                
            except Exception as e:
                logger.error(f"[{job_id}] Chunk {chunk_num} failed: {e}")
                failed_chunks.append({
                    "chunk_num": chunk_num,
                    "start_index": i,
                    "size": len(chunk),
                    "error": str(e)
                })
                # Continue processing other chunks (resilient)
                continue
        
        processing_time = time.time() - start_time
        
        result = {
            "job_id": job_id,
            "status": "completed" if not failed_chunks else "completed_with_errors",
            "total_texts": len(texts),
            "embeddings_count": len(all_embeddings),
            "processing_time_s": round(processing_time, 2),
            "chunks_processed": chunks_processed,
            "failed_chunks": len(failed_chunks),
            "failed_chunk_details": failed_chunks if failed_chunks else None,
        }
        
        logger.info(
            f"[{job_id}] ✅ Batch embedding complete: "
            f"{len(all_embeddings)}/{len(texts)} texts, "
            f"{processing_time:.2f}s"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"[{job_id}] ❌ Batch embedding failed: {e}", exc_info=True)
        
        # Retry on failure
        try:
            raise self.retry(exc=e)
        except self.MaxRetriesExceededError:
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
            }