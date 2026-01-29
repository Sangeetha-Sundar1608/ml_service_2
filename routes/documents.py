"""
Document Routes - Document Processing Endpoints
EPIC 2: Document Processing
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional
import logging
import tempfile
import os
from pathlib import Path

from models.request_models import DocumentFormat
from models.response_models import DocumentParseResponse
from services.document_processor import DocumentProcessor
from main import get_document_processor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/document", response_model=DocumentParseResponse)
async def parse_document(
    file: UploadFile = File(...),
    format_hint: Optional[str] = Form(None),
    extract_metadata: bool = Form(True),
    ocr_enabled: bool = Form(True),
    document_processor: DocumentProcessor = Depends(get_document_processor),
):
    """
    Parse document and extract text
    
    Supports: PDF, DOCX, Images (JPG, PNG, GIF, BMP, TIFF)
    
    **Request:**
    - `file`: Document file (multipart/form-data)
    - `format_hint`: Optional format hint (pdf, docx, image)
    - `extract_metadata`: Extract document metadata
    - `ocr_enabled`: Use OCR for scanned documents/images
    
    **Response:**
```json
    {
      "text": "Extracted text content...",
      "format": "pdf",
      "pages": 5,
      "quality_score": 0.95,
      "language": "en",
      "processing_time_ms": 1200,
      "metadata": {
        "created_date": "2024-01-15",
        "encoding": "utf-8"
      }
    }
```
    """
    temp_path = None
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Get file extension
        file_ext = Path(file.filename).suffix.lower()
        
        if not file_ext:
            raise HTTPException(status_code=400, detail="File has no extension")
        
        # Validate format
        allowed_extensions = {
            ".pdf", ".docx",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"
        }
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported format: {file_ext}. Allowed: {allowed_extensions}"
            )
        
        logger.info(f"Processing document: {file.filename} ({file_ext})")
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as temp_file:
            # Read and write file
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Validate file size
        file_size = os.path.getsize(temp_path)
        max_size = 50 * 1024 * 1024  # 50MB
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size / 1024 / 1024:.1f}MB (max 50MB)"
            )
        
        # Process document
        result = await document_processor.extract(
            file_path=temp_path,
            format_hint=format_hint
        )
        
        logger.info(
            f"✅ Document processed: {result['format']}, "
            f"quality={result.get('quality_score')}, "
            f"time={result['processing_time_ms']}ms"
        )
        
        return DocumentParseResponse(**result)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Document processing failed",
                "message": str(e),
            }
        )
    
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@router.get("/formats")
async def get_supported_formats():
    """
    Get list of supported document formats
    
    **Response:**
```json
    {
      "formats": ["pdf", "docx", "image"],
      "extensions": {
        "pdf": [".pdf"],
        "docx": [".docx"],
        "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
      }
    }
```
    """
    return {
        "formats": ["pdf", "docx", "image"],
        "extensions": {
            "pdf": [".pdf"],
            "docx": [".docx"],
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"]
        },
        "max_file_size_mb": 50,
    }
    