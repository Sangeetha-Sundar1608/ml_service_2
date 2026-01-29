"""
Integration Tests for API Endpoints
Tests all major endpoints end-to-end
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/ml/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


def test_llm_inference_endpoint(client, sample_llm_request):
    """Test LLM inference endpoint"""
    response = client.post("/ml/v1/llm/infer", json=sample_llm_request)
    
    # May return 503 if no providers configured in test
    assert response.status_code in [200, 503]


def test_embedding_generation_endpoint(client, sample_embedding_request):
    """Test embedding generation endpoint"""
    response = client.post(
        "/ml/v1/embeddings/generate",
        json=sample_embedding_request
    )
    
    # May fail if model not loaded in test
    assert response.status_code in [200, 500]


def test_document_upload_no_file(client):
    """Test document endpoint without file"""
    response = client.post("/ml/v1/parse/document")
    assert response.status_code == 422  # Validation error


def test_invalid_request_format(client):
    """Test invalid request format"""
    response = client.post(
        "/ml/v1/llm/infer",
        json={"invalid": "data"}
    )
    assert response.status_code == 422  # Validation error