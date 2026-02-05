"""
Embedding Service - Text Embeddings with OpenAI Fallback
EPIC 3: Text Embeddings (ML-3.1, ML-3.2, ML-3.3)
Generates 384-dimensional semantic vectors with fallback to OpenAI
"""

import logging
import time
from typing import List, Dict, Any, Optional
from functools import lru_cache

from clients.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Unified embedding generation service
    
    Features:
    - Local embeddings (all-MiniLM-L6-v2) - FREE
    - OpenAI fallback (text-embedding-3-small) - PAID
    - Batch processing (32+ texts)
    - L2 normalization
    - Cost tracking
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        openai_client: Optional[OpenAIClient] = None,
        batch_size: int = 32,
    ):
        """
        Initialize embedding service
        
        Args:
            model_name: Local embedding model
            openai_client: OpenAI client for fallback
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.openai_client = openai_client
        self.batch_size = batch_size
        
        # Model loaded lazily on first request
        self._model = None
        self._first_load_time = None
        
        logger.info(
            f"✅ Embedding service initialized: "
            f"model={model_name}, batch_size={batch_size}"
        )
    
    @property
    def model(self):
        """Lazy load model on first access"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()
            
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.error("❌ sentence-transformers not installed. Local embeddings disabled.")
                raise RuntimeError("sentence-transformers library missing. Install it or use OpenAI fallback.")
            
            self._first_load_time = time.time() - start_time
            logger.info(
                f"✅ Model loaded in {self._first_load_time:.2f}s "
                f"(dim={self._model.get_sentence_embedding_dimension()})"
            )
        
        return self._model
    
    async def generate(
        self,
        texts: List[str],
        normalize: bool = True,
        use_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for texts with OpenAI fallback
        
        EPIC 3.1: Local Embedding Generation
        EPIC 3.3: OpenAI Embedding Fallback
        
        Args:
            texts: List of texts to embed
            normalize: L2 normalize embeddings
            use_fallback: Use OpenAI if local fails
        
        Returns:
            {
                "embeddings": [[f1, f2, ...], ...],
                "model": "model_name",
                "provider": "local" or "openai",
                "tokens": 100,
                "dimension": 384,
                "processing_time_ms": 50
            }
        
        Raises:
            RuntimeError: If both local and OpenAI fail
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        
        start_time = time.time()
        
        # Try local embeddings first
        try:
            logger.debug(f"Generating local embeddings for {len(texts)} texts")
            
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Convert to Python lists
            embeddings_list = [[float(x) for x in emb] for emb in embeddings]
            
            # Calculate tokens (approximate)
            tokens = sum(len(text.split()) for text in texts)
            
            logger.info(
                f"✅ Local embeddings generated: {len(texts)} texts, "
                f"{processing_time_ms}ms, ~{processing_time_ms/len(texts):.1f}ms/text"
            )
            
            return {
                "embeddings": embeddings_list,
                "model": self.model_name,
                "provider": "local",
                "tokens": tokens,
                "dimension": len(embeddings_list[0]),
                "processing_time_ms": processing_time_ms,
            }
        
        except Exception as e:
            logger.warning(f"Local embedding failed: {e}")
            
            # Fallback to OpenAI if enabled
            if use_fallback and self.openai_client:
                logger.info("Falling back to OpenAI embeddings")
                return await self._generate_openai(texts)
            else:
                logger.error("No fallback available, embedding failed")
                raise RuntimeError(f"Local embedding failed: {e}")
    
    async def _generate_openai(
        self,
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Generate embeddings using OpenAI (fallback)
        
        EPIC 3.3: OpenAI Embedding Fallback
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Same format as local embeddings
        
        Raises:
            RuntimeError: If OpenAI fails
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI client not configured")
        
        try:
            start_time = time.time()
            
            logger.debug(f"Generating OpenAI embeddings for {len(texts)} texts")
            
            # Call OpenAI client
            embeddings = await self.openai_client.generate_embeddings(
                texts=texts,
                model="text-embedding-3-small"
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate tokens (OpenAI counts differently, use approximation)
            tokens = sum(len(text.split()) for text in texts)
            
            # Calculate cost (done in OpenAI client, but log here)
            cost_per_token = 0.020 / 1_000_000  # $0.020 per 1M tokens
            cost_usd = tokens * cost_per_token
            
            logger.warning(
                f"⚠️ OpenAI embeddings used: {len(texts)} texts, "
                f"{tokens} tokens, cost=${cost_usd:.6f}"
            )
            
            # Alert if cost threshold exceeded (handled by cost tracker)
            # This is just a warning log
            if cost_usd > 0.01:  # > 1 cent per request
                logger.warning(
                    f"⚠️ High OpenAI embedding cost: ${cost_usd:.4f} for {len(texts)} texts"
                )
            
            return {
                "embeddings": embeddings,
                "model": "text-embedding-3-small",
                "provider": "openai",
                "tokens": tokens,
                "dimension": len(embeddings[0]) if embeddings else 0,
                "processing_time_ms": processing_time_ms,
                "cost_usd": cost_usd,
            }
        
        except Exception as e:
            logger.error(f"❌ OpenAI embedding fallback failed: {e}")
            raise RuntimeError(f"OpenAI embedding failed: {e}")
    
    async def generate_batch(
        self,
        texts: List[str],
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for large batch (chunked processing)
        
        EPIC 3.2: Batch Embedding with Optimization
        
        Args:
            texts: List of texts (up to 10,000)
            chunk_size: Process in chunks of this size
        
        Returns:
            {
                "embeddings": [[...], ...],
                "chunks_processed": 10,
                "total_texts": 10000,
                "model": "model_name",
                "provider": "local",
                "total_tokens": 50000,
                "processing_time_ms": 30000
            }
        """
        if not texts:
            raise ValueError("texts cannot be empty")
        
        if len(texts) > 10000:
            raise ValueError("Maximum 10,000 texts allowed")
        
        start_time = time.time()
        total_texts = len(texts)
        
        logger.info(f"Starting batch embedding: {total_texts} texts, chunk_size={chunk_size}")
        
        # Process in chunks
        all_embeddings = []
        chunks_processed = 0
        
        for i in range(0, total_texts, chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            logger.debug(f"Processing chunk {chunk_num}: {len(chunk)} texts")
            
            try:
                # Generate embeddings for this chunk
                result = await self.generate(chunk, normalize=True)
                all_embeddings.extend(result["embeddings"])
                chunks_processed += 1
                
            except Exception as e:
                logger.error(f"Chunk {chunk_num} failed: {e}")
                # Continue processing other chunks (resilient)
                # Fill with zeros for failed chunk
                all_embeddings.extend([[0.0] * 384 for _ in chunk])
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        total_tokens = sum(len(text.split()) for text in texts)
        
        logger.info(
            f"✅ Batch embedding complete: {total_texts} texts, "
            f"{chunks_processed} chunks, {processing_time_ms}ms"
        )
        
        return {
            "embeddings": all_embeddings,
            "chunks_processed": chunks_processed,
            "total_texts": total_texts,
            "model": self.model_name,
            "provider": "local",
            "total_tokens": total_tokens,
            "processing_time_ms": processing_time_ms,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            {
                "name": "all-MiniLM-L6-v2",
                "loaded": True,
                "dimension": 384,
                "first_load_time_s": 2.5
            }
        """
        return {
            "name": self.model_name,
            "loaded": self._model is not None,
            "dimension": self._model.get_sentence_embedding_dimension() if self._model else 384,
            "first_load_time_s": self._first_load_time if self._first_load_time else 0.0,
        }