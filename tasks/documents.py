"""
Celery Tasks - Batch Document Processing
EPIC 2: Document Processing
Async processing of document batches
"""

import logging
import time
from typing import List, Dict, Any
from pathlib import Path
from celery import Task

from celery_app import celery_app
from services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class DocumentTask(Task):
    """Base task with shared document processor"""
    _document_processor = None
    
    @property
    def document_processor(self) -> DocumentProcessor:
        if self._document_processor is None:
            self._document_processor = DocumentProcessor()
        return self._document_processor


@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="tasks.process_batch_documents",
    max_retries=3,
    default_retry_delay=60,
)
def process_batch_documents(
    self,
    job_id: str,
    file_paths: List[str],
) -> Dict[str, Any]:
    """
    Process batch documents asynchronously
    
    Args:
        job_id: Unique job identifier
        file_paths: List of file paths to process
    
    Returns:
        {
            "job_id": "uuid",
            "status": "completed",
            "total_documents": 100,
            "processed": 98,
            "failed": 2,
            "processing_time_s": 120.5
        }
    """
    logger.info(f"[{job_id}] Starting batch document processing: {len(file_paths)} files")
    
    start_time = time.time()
    
    try:
        results = []
        failed = []
        
        for idx, file_path in enumerate(file_paths, 1):
            logger.info(f"[{job_id}] Processing document {idx}/{len(file_paths)}: {file_path}")
            
            try:
                # Process document
                import asyncio
                result = asyncio.run(
                    self.document_processor.extract(file_path)
                )
                
                results.append({
                    "file": file_path,
                    "success": True,
                    "format": result.get("format"),
                    "text_length": len(result.get("text", "")),
                })
                
                # Update progress
                progress = int((idx / len(file_paths)) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': idx,
                        'total': len(file_paths),
                        'progress': progress,
                    }
                )
                
            except Exception as e:
                logger.error(f"[{job_id}] Document {file_path} failed: {e}")
                failed.append({
                    "file": file_path,
                    "error": str(e)
                })
                continue
        
        processing_time = time.time() - start_time
        
        result = {
            "job_id": job_id,
            "status": "completed" if not failed else "completed_with_errors",
            "total_documents": len(file_paths),
            "processed": len(results),
            "failed": len(failed),
            "processing_time_s": round(processing_time, 2),
            "failed_documents": failed if failed else None,
        }
        
        logger.info(
            f"[{job_id}] ✅ Batch document processing complete: "
            f"{len(results)}/{len(file_paths)} processed, {processing_time:.2f}s"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"[{job_id}] ❌ Batch document processing failed: {e}", exc_info=True)
        
        try:
            raise self.retry(exc=e)
        except self.MaxRetriesExceededError:
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
            }