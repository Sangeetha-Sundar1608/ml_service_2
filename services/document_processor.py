"""
Document Processor - Unified Document Extraction Service
EPIC 2: Document Processing
Handles PDF, DOCX, and OCR extraction
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

# Import your existing processors
from services.pdf_processor import PDFProcessor
from services.docx_processor import DOCXProcessor
from services.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)


class DocumentFormat(str, Enum):
    """Supported document formats"""
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    UNKNOWN = "unknown"


class DocumentProcessor:
    """
    Unified document processing service
    
    Routes documents to appropriate processor:
    - PDF → PDFProcessor
    - DOCX → DOCXProcessor
    - Images → OCRProcessor
    """
    
    def __init__(self):
        """Initialize all document processors"""
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()
        self.ocr_processor = OCRProcessor()
        
        logger.info("✅ Document processor initialized (PDF, DOCX, OCR)")
    
    async def extract(
        self,
        file_path: str,
        format_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract text from document
        
        Args:
            file_path: Path to document file
            format_hint: Optional format hint (pdf, docx, image)
        
        Returns:
            Extraction result (format depends on document type)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If unsupported format
        """
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect format
        doc_format = self._detect_format(path, format_hint)
        
        logger.info(f"Processing {doc_format.value} document: {file_path}")
        
        # Route to appropriate processor
        if doc_format == DocumentFormat.PDF:
            result = await self.pdf_processor.extract(file_path)
            return result.dict()
        
        elif doc_format == DocumentFormat.DOCX:
            result = await self.docx_processor.extract(file_path)
            return result.dict()
        
        elif doc_format == DocumentFormat.IMAGE:
            result = await self.ocr_processor.extract(file_path)
            return result.dict()
        
        else:
            raise ValueError(
                f"Unsupported document format: {path.suffix}. "
                f"Supported: .pdf, .docx, .jpg, .png, .gif, .bmp, .tiff"
            )
    
    def _detect_format(
        self,
        path: Path,
        format_hint: Optional[str] = None
    ) -> DocumentFormat:
        """
        Detect document format from extension
        
        Args:
            path: File path
            format_hint: Optional format override
        
        Returns:
            DocumentFormat enum
        """
        # Use hint if provided
        if format_hint:
            try:
                return DocumentFormat(format_hint.lower())
            except ValueError:
                logger.warning(f"Invalid format hint: {format_hint}, using auto-detect")
        
        # Detect from extension
        ext = path.suffix.lower()
        
        if ext == ".pdf":
            return DocumentFormat.PDF
        
        elif ext == ".docx":
            return DocumentFormat.DOCX
        
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"]:
            return DocumentFormat.IMAGE
        
        else:
            return DocumentFormat.UNKNOWN
        