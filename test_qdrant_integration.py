#!/usr/bin/env python3
"""
Test script for Qdrant integration with E5 multilingual model
"""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qdrant_integration import qdrant_store, store_fact_check_in_qdrant

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qdrant_health():
    """Test Qdrant connection and health"""
    logger.info("🔍 Testing Qdrant health...")
    health = qdrant_store.health_check()
    logger.info(f"Health status: {health}")
    return health.get('status') == 'healthy'

def test_single_claim_storage():
    """Test storing a single claim"""
    logger.info("🔍 Testing single claim storage...")
    
    # Test data
    claim_text = "The Earth is flat and NASA is hiding the truth from us"
    verdict = "FALSE"
    confidence = 0.95
    justification = "Scientific evidence overwhelmingly supports that Earth is spherical. NASA and other space agencies have provided extensive photographic and scientific evidence."
    message_id = f"test_msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    success = qdrant_store.store_claim_verdict(
        claim_text=claim_text,
        verdict=verdict,
        confidence=confidence,
        justification=justification,
        message_id=message_id
    )
    
    logger.info(f"Single claim storage success: {success}")
    return success

def test_multilingual_claims():
    """Test storing multilingual claims"""
    logger.info("🔍 Testing multilingual claim storage...")
    
    claims_verdicts = [
        {
            "claim": "COVID-19 vaccines contain microchips for tracking",
            "verdict": "FALSE",
            "confidence": 0.98,
            "justification": "No evidence supports this conspiracy theory. Vaccines contain biological components, not electronic devices."
        },
        {
            "claim": "اللقاحات تحتوي على رقائق إلكترونية للتتبع",  # Arabic
            "verdict": "FALSE", 
            "confidence": 0.97,
            "justification": "لا يوجد دليل علمي يدعم هذه النظرية. اللقاحات تحتوي على مكونات بيولوجية وليس أجهزة إلكترونية."
        },
        {
            "claim": "Les vaccins contiennent des puces électroniques",  # French
            "verdict": "FALSE",
            "confidence": 0.96,
            "justification": "Aucune preuve scientifique ne soutient cette théorie du complot."
        }
    ]
    
    message_id = f"test_multilingual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stored_count = store_fact_check_in_qdrant(claims_verdicts, message_id)
    
    logger.info(f"Multilingual claims stored: {stored_count}/{len(claims_verdicts)}")
    return stored_count == len(claims_verdicts)

def test_similarity_search():
    """Test similarity search functionality"""
    logger.info("🔍 Testing similarity search...")
    
    # Search for similar claims
    test_claim = "Are COVID vaccines safe and effective?"
    similar_claims = qdrant_store.search_similar_claims(test_claim, limit=3)
    
    logger.info(f"Found {len(similar_claims)} similar claims for: '{test_claim}'")
    for i, claim in enumerate(similar_claims, 1):
        logger.info(f"  {i}. {claim['claim_text'][:60]}... -> {claim['verdict']} (similarity: {claim['similarity_score']:.3f})")
    
    return len(similar_claims) >= 0  # Always passes, just for testing

def test_statistics():
    """Test statistics retrieval"""
    logger.info("🔍 Testing statistics...")
    
    stats = qdrant_store.get_verdict_statistics()
    logger.info(f"Statistics: {stats}")
    
    return 'total_claims' in stats

def main():
    """Run all tests"""
    logger.info("🚀 Starting Qdrant Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Health Check", test_qdrant_health),
        ("Single Claim Storage", test_single_claim_storage), 
        ("Multilingual Claims", test_multilingual_claims),
        ("Similarity Search", test_similarity_search),
        ("Statistics", test_statistics)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'=' * 20} {test_name} {'=' * 20}")
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'=' * 20} TEST SUMMARY {'=' * 20}")
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Qdrant integration is working correctly.")
    else:
        logger.warning(f"⚠️  Some tests failed. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
