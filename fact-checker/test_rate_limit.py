#!/usr/bin/env python3
"""Simple test with just claim extraction to check rate limiting."""

import asyncio
from claim_extractor import ClaimExtractor

async def test_rate_limiting():
    """Test that rate limiting is working."""
    
    test_text = """The highways leading into Tehran are busy again, filled with cars carrying families, suitcases, and the cautious hope that home might finally be safe. After 12 days of war that killed more than 600 Iranians and displaced hundreds of thousands from the capital, a ceasefire announced on Monday has begun drawing residents back to a city still scarred by Israeli air strikes.

For many returning to Tehran, the relief of sleeping in their own beds is tempered by the constant fear that the bombing could resume at any moment."""
    
    print("Testing claim extraction with rate limiting...")
    
    extractor = ClaimExtractor()
    claims = await extractor.extract_claims(test_text)
    
    print(f"✅ Successfully extracted {len(claims)} claims:")
    for i, claim in enumerate(claims, 1):
        print(f"  {i}. {claim}")

if __name__ == "__main__":
    asyncio.run(test_rate_limiting())
