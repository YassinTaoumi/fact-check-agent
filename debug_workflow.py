#!/usr/bin/env python3
"""
Debug workflow to identify the issue with state handling.
"""

import asyncio
import logging
import sys
import os

# Add the fact-checker directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fact-checker'))

from fact_checker import FactChecker, FactCheckState

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_workflow():
    """Debug the workflow to understand state handling."""
    
    # Simple test text
    test_text = "The Earth is round and orbits the Sun."
    
    print(f"Testing fact-checker with: {test_text}")
    print("=" * 60)
    
    try:
        # Initialize fact checker
        checker = FactChecker()
        print(f"✓ FactChecker initialized")
        
        # Create initial state manually to test
        initial_state = FactCheckState(
            input_text=test_text,
            errors=[],
            workflow_metadata={
                'start_time': '2024-01-01T00:00:00',
                'processing_time': {},
                'steps_completed': []
            }
        )
        
        print(f"✓ Initial state created: {type(initial_state)}")
        print(f"  - input_text: {initial_state.input_text}")
        print(f"  - claims: {initial_state.claims}")
        print(f"  - verdicts: {initial_state.verdicts}")
        
        # Test just the first node manually
        print("\nTesting extract_claims_node manually...")
        try:
            result_state = await checker._extract_claims_node(initial_state)
            print(f"✓ Extract claims result: {type(result_state)}")
            print(f"  - claims: {result_state.claims}")
            print(f"  - errors: {result_state.errors}")
            
        except Exception as e:
            print(f"✗ Error in extract_claims_node: {e}")
            return
        
        # Test the full workflow
        print("\nTesting full workflow...")
        try:
            final_state = await checker.workflow.ainvoke(initial_state)
            print(f"✓ Workflow completed")
            print(f"  - Result type: {type(final_state)}")
            
            if isinstance(final_state, dict):
                print(f"  - Dict keys: {list(final_state.keys())}")
                print(f"  - Has verdicts: {'verdicts' in final_state}")
                if 'verdicts' in final_state:
                    print(f"  - Verdicts type: {type(final_state['verdicts'])}")
            else:
                print(f"  - Has verdicts attr: {hasattr(final_state, 'verdicts')}")
                if hasattr(final_state, 'verdicts'):
                    print(f"  - Verdicts: {final_state.verdicts}")
                    print(f"  - Claims: {final_state.claims}")
            
        except Exception as e:
            print(f"✗ Error in full workflow: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test the full fact-check method
        print("\nTesting full fact-check method...")
        try:
            results = await checker.check_facts(test_text)
            print(f"✓ Fact-check completed")
            print(f"  - Success: {results['success']}")
            print(f"  - Claims count: {len(results['claims'])}")
            print(f"  - Results count: {len(results['results'])}")
            
        except Exception as e:
            print(f"✗ Error in check_facts: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"✗ Error initializing fact checker: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_workflow())
