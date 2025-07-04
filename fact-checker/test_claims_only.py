#!/usr/bin/env python3
"""Test just the claim extractor with Tehran text."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from claim_extractor import ClaimExtractor

def test_claim_extraction():
    test_text = """The highways leading into Tehran are busy again, filled with cars carrying families, suitcases, and the cautious hope that home might finally be safe. After 12 days of war that killed more than 600 Iranians and displaced hundreds of thousands from the capital, a ceasefire announced on Monday has begun drawing residents back to a city still scarred by Israeli air strikes.

For many returning to Tehran, the relief of sleeping in their own beds is tempered by the constant fear that the bombing could resume at any moment."""
    
    print("Testing Claim Extractor with Tehran conflict text...")
    print(f"Text length: {len(test_text)} characters")
    print(f"Text: {test_text}")
    print("\n" + "="*60)
    
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(test_text)
    
    print(f"📋 EXTRACTED {len(claims)} CLAIMS:")
    print("="*60)
    
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim}")
    
    print("\n" + "="*60)
    print("Analysis complete!")

if __name__ == "__main__":
    test_claim_extraction()
