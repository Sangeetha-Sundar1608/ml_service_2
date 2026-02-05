"""
OCR Processor - Extract text from images using Tesseract
"""

import logging
from typing import Dict, Any, Optional
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logger = logging.getLogger(__name__)


class OCRExtractionResult:
    def __init__(self, text: str, confidence: float, language: str = "en", format: str = "image"):
        self.text = text
        self.confidence = confidence
        self.language = language
        self.format = format

    def dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "format": self.format
        }


class OCRProcessor:
    async def extract(self, file_path: str) -> OCRExtractionResult:
        """Extract text from image via OCR"""
        if not HAS_OCR:
            logger.warning("OCR libraries not installed (pytesseract/Pillow). Returning empty text.")
            return OCRExtractionResult(text="", confidence=0.0)

        try:
            # Load image
            image = Image.open(file_path)
            
            # OCR
            # Note: Tesseract must be installed on the system for this to work
            text = pytesseract.image_to_string(image)
            
            # Get confidence (dummy value for MVP unless we use image_to_data)
            confidence = 0.9
            
            return OCRExtractionResult(
                text=text,
                confidence=confidence,
                language="en", # Auto-detect in future
                format="image"
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
