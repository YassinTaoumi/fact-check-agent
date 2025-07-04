"""
Qdrant Vector Database Integration for Fact-Check Results

This module handles:
1. Storing fact-check results as vectors in Qdrant
2. Semantic search for similar claims
3. Knowledge base building from claim verdicts
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Install with: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fact_check_claims")

class QdrantFactCheckStore:
    """Qdrant vector store for fact-check results"""
    
    def __init__(self):
        """Initialize Qdrant client and embedding model"""
        self.client = None
        self.encoder = None
        self.collection_name = QDRANT_COLLECTION
        self.vector_size = 1024  # multilingual-e5-large-instruct embedding size
        
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant client not available")
            return
        
        if not EMBEDDING_AVAILABLE:
            logger.error("SentenceTransformers not available")
            return
        
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            
            # Initialize multilingual E5 large instruct model for high-quality embeddings
            logger.info("🔄 Loading multilingual-e5-large-instruct model...")
            self.encoder = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
            self.vector_size = self.encoder.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Loaded embedding model with dimension: {self.vector_size}")
            
            # Create collection if it doesn't exist
            self._ensure_collection_exists()
            
            logger.info(f"✅ Qdrant fact-check store initialized on {QDRANT_HOST}:{QDRANT_PORT}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant: {e}")
            self.client = None
            self.encoder = None
    
    def _ensure_collection_exists(self):
        """Create the fact-check collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"✅ Collection {self.collection_name} created successfully")
            else:
                logger.info(f"✅ Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"❌ Failed to ensure collection exists: {e}")
            raise
    
    def store_claim_verdict(self, claim_text: str, verdict: str, confidence: float, 
                           justification: str, message_id: str, timestamp: Optional[str] = None) -> bool:
        """
        Store a claim verdict in Qdrant vector database
        
        Args:
            claim_text: The text of the claim
            verdict: The verdict (TRUE, FALSE, PARTLY_TRUE, UNVERIFIED)
            confidence: Confidence score (0.0-1.0)
            justification: Reasoning for the verdict
            message_id: Original message ID
            timestamp: ISO timestamp (defaults to now)
            
        Returns:
            bool: Success status
        """
        if not self.client or not self.encoder:
            logger.warning("Qdrant store not available, skipping vector storage")
            return False
        
        try:
            # Prepare the text for E5 embedding with instruction format
            # E5 models work better with instruction prefixes for different tasks
            instruct_text = f"query: {claim_text}"
            
            # Generate embedding for the claim using E5 instruct model
            embedding = self.encoder.encode(instruct_text, normalize_embeddings=True).tolist()
            
            # Generate unique point ID
            point_id = str(uuid.uuid4())
            
            # Prepare timestamp
            if not timestamp:
                timestamp = datetime.now().isoformat()
            
            # Create payload with metadata including the embedding info
            payload = {
                # Core claim data
                "claim_text": claim_text,
                "verdict": verdict,
                "confidence": confidence,
                "justification": justification,
                "message_id": message_id,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                
                # Embedding metadata
                "embedding_model": "intfloat/multilingual-e5-large-instruct",
                "embedding_dimension": len(embedding),
                "instruct_text": instruct_text,  # The actual text that was embedded
                
                # Additional searchable fields
                "verdict_category": verdict.lower(),
                "confidence_level": self._get_confidence_level(confidence),
                "claim_length": len(claim_text),
                "has_justification": len(justification) > 0,
                "language_detected": self._detect_language_hint(claim_text)
            }
            
            # Store in Qdrant
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"✅ Stored claim verdict in Qdrant: {claim_text[:50]}... -> {verdict} (confidence: {confidence:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store claim in Qdrant: {e}")
            return False
    
    def store_multiple_claims(self, claims_verdicts: List[Dict[str, Any]], message_id: str) -> int:
        """
        Store multiple claim verdicts from a single fact-check session
        
        Args:
            claims_verdicts: List of claim verdict dictionaries
            message_id: Original message ID
            
        Returns:
            int: Number of claims successfully stored
        """
        if not self.client or not self.encoder:
            logger.warning("Qdrant store not available, skipping vector storage")
            return 0
        
        stored_count = 0
        timestamp = datetime.now().isoformat()
        
        for claim_verdict in claims_verdicts:
            try:
                claim_text = claim_verdict.get('claim', '')
                verdict = claim_verdict.get('verdict', 'UNVERIFIED')
                confidence = claim_verdict.get('confidence', 0.0)
                justification = claim_verdict.get('justification', '')
                
                if claim_text:  # Only store if claim text exists
                    success = self.store_claim_verdict(
                        claim_text=claim_text,
                        verdict=verdict,
                        confidence=confidence,
                        justification=justification,
                        message_id=message_id,
                        timestamp=timestamp
                    )
                    if success:
                        stored_count += 1
                        
            except Exception as e:
                logger.error(f"❌ Failed to store individual claim: {e}")
                continue
        
        logger.info(f"✅ Stored {stored_count}/{len(claims_verdicts)} claims in Qdrant")
        return stored_count
    
    def search_similar_claims(self, claim_text: str, limit: int = 5, 
                             min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar claims in the vector database
        
        Args:
            claim_text: The claim to search for
            limit: Maximum number of results
            min_confidence: Minimum confidence threshold for results
            
        Returns:
            List of similar claims with metadata
        """
        if not self.client or not self.encoder:
            logger.warning("Qdrant store not available")
            return []
        
        try:
            # Generate embedding for search query
            query_embedding = self.encoder.encode(claim_text).tolist()
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.7  # Cosine similarity threshold
            )
            
            # Format results
            similar_claims = []
            for result in search_results:
                payload = result.payload
                
                # Filter by confidence if specified
                if payload.get('confidence', 0) >= min_confidence:
                    similar_claims.append({
                        'claim_text': payload.get('claim_text', ''),
                        'verdict': payload.get('verdict', ''),
                        'confidence': payload.get('confidence', 0.0),
                        'justification': payload.get('justification', ''),
                        'similarity_score': result.score,
                        'timestamp': payload.get('timestamp', ''),
                        'message_id': payload.get('message_id', '')
                    })
            
            logger.info(f"Found {len(similar_claims)} similar claims for: {claim_text[:50]}...")
            return similar_claims
            
        except Exception as e:
            logger.error(f"❌ Failed to search similar claims: {e}")
            return []
    
    def get_verdict_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored verdicts"""
        if not self.client:
            return {}
        
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            
            # Get verdict distribution using scroll
            verdict_counts = {"TRUE": 0, "FALSE": 0, "PARTLY_TRUE": 0, "UNVERIFIED": 0}
            confidence_scores = []
            
            # Scroll through all points (in batches)
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Batch size
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            
            for point in points:
                verdict = point.payload.get('verdict', 'UNVERIFIED')
                confidence = point.payload.get('confidence', 0.0)
                
                if verdict in verdict_counts:
                    verdict_counts[verdict] += 1
                
                confidence_scores.append(confidence)
            
            # Calculate statistics
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            return {
                'total_claims': total_points,
                'verdict_distribution': verdict_counts,
                'average_confidence': avg_confidence,
                'high_confidence_claims': len([c for c in confidence_scores if c >= 0.8]),
                'collection_size_mb': collection_info.config.params.vectors.size * total_points * 4 / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {}
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to categorical level"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def _detect_language_hint(self, text: str) -> str:
        """
        Simple language detection based on character patterns
        
        Args:
            text: Input text to analyze
            
        Returns:
            str: Language hint (en, ar, fr, es, etc.) or 'unknown'
        """
        if not text:
            return "unknown"
        
        # Simple heuristics for common languages
        # Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        if arabic_chars > len(text) * 0.3:
            return "ar"
        
        # French indicators
        if any(word in text.lower() for word in ['le ', 'la ', 'les ', 'une ', 'des ', 'est ', 'sont ']):
            return "fr"
        
        # Spanish indicators  
        if any(word in text.lower() for word in ['el ', 'la ', 'los ', 'las ', 'una ', 'es ', 'son ']):
            return "es"
        
        # Default to English for Latin scripts
        if all(ord(char) < 256 for char in text):
            return "en"
        
        return "multilingual"
    
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant connection and collection status"""
        if not self.client:
            return {
                "status": "unavailable",
                "error": "Qdrant client not initialized"
            }
        
        try:
            # Test connection
            collections = self.client.get_collections()
            
            # Check if our collection exists
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if collection_exists:
                collection_info = self.client.get_collection(self.collection_name)
                return {
                    "status": "healthy",
                    "collection_exists": True,
                    "points_count": collection_info.points_count,
                    "vector_size": self.vector_size,
                    "host": QDRANT_HOST,
                    "port": QDRANT_PORT
                }
            else:
                return {
                    "status": "collection_missing",
                    "collection_exists": False,
                    "host": QDRANT_HOST,
                    "port": QDRANT_PORT
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Global instance
qdrant_store = QdrantFactCheckStore()

def store_fact_check_in_qdrant(claims_verdicts: List[Dict[str, Any]], message_id: str) -> int:
    """
    Convenience function to store fact-check results in Qdrant
    
    Args:
        claims_verdicts: List of claim verdict dictionaries
        message_id: Original message ID
        
    Returns:
        int: Number of claims stored
    """
    return qdrant_store.store_multiple_claims(claims_verdicts, message_id)

def search_similar_fact_checks(claim_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to search for similar claims
    
    Args:
        claim_text: The claim to search for
        limit: Maximum number of results
        
    Returns:
        List of similar claims
    """
    return qdrant_store.search_similar_claims(claim_text, limit)
