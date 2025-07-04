#!/usr/bin/env python3
"""Test the overall verdict generation with mock data."""

import asyncio
from verdict_analyzer import VerdictAnalyzer

async def test_overall_verdict():
    """Test overall verdict generation with sample data."""
    
    input_text = """The highways leading into Tehran are busy again, filled with cars carrying families, suitcases, and the cautious hope that home might finally be safe. After 12 days of war that killed more than 600 Iranians and displaced hundreds of thousands from the capital, a ceasefire announced on Monday has begun drawing residents back to a city still scarred by Israeli air strikes.

For many returning to Tehran, the relief of sleeping in their own beds is tempered by the constant fear that the bombing could resume at any moment."""
    
    claims = [
        "The war between Iran and Israel lasted for 12 days.",
        "More than 600 Iranians were killed in the conflict.",
        "Hundreds of thousands of people were displaced from the capital, Tehran."
    ]
    
    # Mock individual verdicts (simulating what would come from web search)
    individual_verdicts = {
        "The war between Iran and Israel lasted for 12 days.": {
            "verdict": "UNVERIFIED",
            "confidence": 0.3,
            "justification": "No recent credible sources found confirming a 12-day Iran-Israel conflict."
        },
        "More than 600 Iranians were killed in the conflict.": {
            "verdict": "UNVERIFIED",
            "confidence": 0.2,
            "justification": "Cannot verify specific casualty numbers from reliable sources."
        },
        "Hundreds of thousands of people were displaced from the capital, Tehran.": {
            "verdict": "UNVERIFIED",
            "confidence": 0.2,
            "justification": "No evidence found of mass displacement from Tehran in recent conflicts."
        }
    }
    
    print("Testing Overall Verdict Generation...")
    print(f"Input text length: {len(input_text)} characters")
    print(f"Claims to analyze: {len(claims)}")
    
    # Test the overall verdict generation
    analyzer = VerdictAnalyzer()
    overall_verdict = await analyzer.generate_overall_verdict(input_text, claims, individual_verdicts)
    
    print("\n" + "="*80)
    print("OVERALL VERDICT RESULTS")
    print("="*80)
    
    print(f"🏆 OVERALL VERDICT: {overall_verdict.get('overall_verdict', 'Unknown')}")
    print(f"🎯 CONFIDENCE: {overall_verdict.get('confidence', 0):.2f}")
    print(f"📄 JUSTIFICATION: {overall_verdict.get('justification', 'No justification')}")
    
    breakdown = overall_verdict.get('claim_breakdown', {})
    if breakdown:
        print(f"\n📊 CLAIM BREAKDOWN:")
        print(f"   ✅ True claims: {breakdown.get('true_claims', 0)}")
        print(f"   ❌ False claims: {breakdown.get('false_claims', 0)}")
        print(f"   ❓ Unverified claims: {breakdown.get('unverified_claims', 0)}")
        print(f"   📝 Total claims: {breakdown.get('total_claims', 0)}")
    
    methodology = overall_verdict.get('methodology', '')
    if methodology:
        print(f"\n🔬 METHODOLOGY: {methodology}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test_overall_verdict())
