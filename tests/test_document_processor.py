"""
Tests for Document Processor
EPIC 2: Document Processing Testing
"""

import pytest
from pathlib import Path


@pytest.mark.asyncio
async def test_pdf_extraction(sample_document_file):
    """Test PDF text extraction"""
    from services.document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    result = await processor.extract(str(sample_document_file))
    
    assert result["format"] == "pdf"
    assert "text" in result
    assert result["processing_time_ms"] > 0


@pytest.mark.asyncio
async def test_unsupported_format():
    """Test error on unsupported format"""
    from services.document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    
    with pytest.raises(ValueError, match="Unsupported"):
        await processor.extract("test.xyz")


@pytest.mark.asyncio
async def test_file_not_found():
    """Test error when file doesn't exist"""
    from services.document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    
    with pytest.raises(FileNotFoundError):
        await processor.extract("nonexistent.pdf")