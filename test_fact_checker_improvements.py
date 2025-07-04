#!/usr/bin/env python3
"""
Quick test of fact-checking improvements
"""

import sys
import os
import asyncio

# Add fact-checker to path
fact_checker_path = os.path.join(os.path.dirname(__file__), 'fact-checker')
if fact_checker_path not in sys.path:
    sys.path.append(fact_checker_path)

async def test_fact_checker():
    """Test the improved fact-checker with the Iran-Israel claim"""
    try:
        from fact_checker import fact_check_text
        
        # Test claim about Iran attacking Israel's defense minister
        test_claim = "COVID 19 vaccines contain microchips for tracking"
        
        print("🧪 Testing improved fact-checker...")
        print(f"📝 Claim: {test_claim}")
        print("🔄 Processing...")
        
        # Run fact-checking
        result = await fact_check_text(test_claim)
        
        print("\n✅ Fact-check completed!")
        print("=" * 50)
        
        if result.get('success'):
            overall_stats = result.get('overall_statistics', {})
            print(f"🎯 Overall Verdict: {overall_stats.get('overall_verdict', 'UNKNOWN')}")
            print(f"📊 Confidence: {overall_stats.get('overall_confidence', 0.0):.1%}")
            print(f"💭 Reasoning: {overall_stats.get('overall_reasoning', 'No reasoning')}")
            
            # Show key sources referenced
            key_sources = overall_stats.get('key_sources_referenced', [])
            if key_sources:
                print("\n🔗 Key Sources Referenced:")
                for i, source in enumerate(key_sources[:5], 1):
                    print(f"  {i}. {source}")
            
            # Show individual claim results
            results = result.get('results', [])
            print(f"\n🔍 Individual Claims ({len(results)}):")
            for i, claim_result in enumerate(results, 1):
                print(f"  {i}. {claim_result.get('claim', '')}")
                print(f"     Verdict: {claim_result.get('verdict', 'UNKNOWN')}")
                print(f"     Confidence: {claim_result.get('confidence', 0.0):.1%}")
                print(f"     Justification: {claim_result.get('justification', 'No justification')[:150]}...")
                
                # Show supporting and contradicting sources
                supporting_sources = claim_result.get('supporting_sources', [])
                if supporting_sources:
                    print(f"     ✅ Supporting Sources:")
                    for j, source in enumerate(supporting_sources[:3], 1):
                        print(f"       {j}. {source}")
                
                contradicting_sources = claim_result.get('contradicting_sources', [])
                if contradicting_sources:
                    print(f"     ❌ Contradicting Sources:")
                    for j, source in enumerate(contradicting_sources[:3], 1):
                        print(f"       {j}. {source}")
                
                print()
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_fact_checker())
    
    if result and result.get('success'):
        print("🎉 Improved fact-checker is working!")
    else:
        print("⚠️ Fact-checker needs more work")
