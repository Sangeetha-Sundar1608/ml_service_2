"""
PDF Processor - Extract text and metadata from PDF files
"""

import logging
import io
from typing import Dict, Any, Optional
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFExtractionResult:
    def __init__(self, text: str, pages: int, quality_score: float, format: str = "pdf"):
        self.text = text
        self.pages = pages
        self.quality_score = quality_score
        self.format = format

    def dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "pages": self.pages,
            "quality_score": self.quality_score,
            "format": self.format
        }


class PDFProcessor:
    async def extract(self, file_path: str) -> PDFExtractionResult:
        """Extract text from PDF"""
        try:
            # Load PDF
            reader = PdfReader(file_path)
            
            # Extract text from all pages
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    # Page might be image-only
                    logger.warning(f"Page {page_num} empty - possibly scanned")
                    text_parts.append("")
                else:
                    text_parts.append(text)
            
            full_text = "\n".join(text_parts)
            
            # Calculate quality score (simple heuristic: ratio of printable chars)
            quality = self._calculate_quality(full_text)
            
            return PDFExtractionResult(
                text=full_text,
                pages=len(reader.pages),
                quality_score=quality,
                format="pdf"
            )
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

    def _calculate_quality(self, text: str) -> float:
        """Calculate a rough quality score for the extraction"""
        if not text:
            return 0.0
        
        # Count valid characters vs total
        total = len(text)
        if total == 0:
            return 0.0
            
        return 1.0  # Simplified for MVP
