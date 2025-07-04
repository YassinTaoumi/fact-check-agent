"""
Claim extraction component for the fact-checker system.
"""

import json
import re
import logging
import asyncio
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from config import config, CLAIM_EXTRACTION_PROMPT
from rate_limiter import llm_rate_limiter

logger = logging.getLogger(__name__)

class ClaimExtractor:
    """Extract factual claims from text using AI."""
    
    def __init__(self):
        """Initialize the claim extractor."""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=config.google_ai_api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        logger.info("ClaimExtractor initialized with Google AI")
    
    async def extract_claims(self, text: str, language: str = "en") -> List[str]:
        """
        Extract factual claims from the given text.
        
        Args:
            text: Input text to analyze
            language: ISO language code (en, ar, fr, es, etc.)
            
        Returns:
            List of extracted claims
        """
        try:
            if not text or len(text.strip()) < config.min_claim_length:
                logger.warning("Text too short for claim extraction")
                return []
            
            # Import language utilities here to avoid circular imports
            from language_utils import get_language_templates, prepare_prompt
            
            # Get language-specific templates
            lang_templates = get_language_templates(language)
            
            # Prepare the prompt
            prompt = prepare_prompt(
                CLAIM_EXTRACTION_PROMPT,
                language,
                text=text,
                max_claims=config.max_claims_per_text,
                min_length=config.min_claim_length
            )
            
            logger.info(f"Extracting claims from text of length: {len(text)}")
            
            # Get AI response
            messages = [
                SystemMessage(content="You are a professional fact-checking assistant."),
                HumanMessage(content=prompt)
            ]
            
            await llm_rate_limiter.wait_if_needed()
            response = self.llm(messages)
            
            # Parse the JSON response
            claims = self._parse_claims_response(response.content)
            
            # Filter and validate claims
            filtered_claims = self._filter_claims(claims)
            
            logger.info(f"Extracted {len(filtered_claims)} claims from text")
            return filtered_claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []
    
    def _parse_claims_response(self, response: str) -> List[str]:
        """Parse the AI response to extract claims."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*"claims".*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return data.get("claims", [])
            
            # Fallback: try to parse the entire response as JSON
            data = json.loads(response)
            return data.get("claims", [])
            
        except json.JSONDecodeError:
            logger.error("Failed to parse claims JSON response")
            # Fallback: extract numbered claims
            return self._extract_numbered_claims(response)
    
    def _extract_numbered_claims(self, response: str) -> List[str]:
        """Fallback method to extract numbered claims from response."""
        claims = []
        lines = response.split('\n')
        
        for line in lines:
            # Look for numbered claims (1., 2., etc.)
            match = re.match(r'^\d+\.\s*(.+)', line.strip())
            if match:
                claim = match.group(1).strip()
                if len(claim) >= config.min_claim_length:
                    claims.append(claim)
        
        return claims
    
    def _filter_claims(self, claims: List[str]) -> List[str]:
        """Filter and validate extracted claims."""
        filtered = []
        
        for claim in claims:
            if not isinstance(claim, str):
                continue
                
            claim = claim.strip()
            
            # Check minimum length
            if len(claim) < config.min_claim_length:
                continue
            
            # Remove quotes if present
            if claim.startswith('"') and claim.endswith('"'):
                claim = claim[1:-1]
            
            # Skip opinion indicators
            opinion_indicators = [
                "i think", "i believe", "in my opinion", "i feel",
                "maybe", "perhaps", "possibly", "might be",
                "could be", "seems like", "appears to"
            ]
            
            if any(indicator in claim.lower() for indicator in opinion_indicators):
                continue
            
            # Skip questions
            if claim.endswith('?'):
                continue
            
            # Skip very short claims
            if len(claim.split()) < 3:
                continue
            
            filtered.append(claim)
            
            # Limit number of claims
            if len(filtered) >= config.max_claims_per_text:
                break
        
        return filtered
    
    def batch_extract_claims(self, texts: List[str]) -> Dict[int, List[str]]:
        """
        Extract claims from multiple texts.
        
        Args:
            texts: List of texts to process
            
        Returns:
            Dictionary mapping text index to extracted claims
        """
        results = {}
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            claims = self.extract_claims(text)
            results[i] = claims
        
        return results
    
    def validate_claim(self, claim: str) -> bool:
        """
        Validate if a claim is suitable for fact-checking.
        
        Args:
            claim: Claim to validate
            
        Returns:
            True if claim is valid for fact-checking
        """
        if not claim or len(claim.strip()) < config.min_claim_length:
            return False
        
        # Check for factual nature (very basic heuristics)
        factual_indicators = [
            "is", "was", "are", "were", "has", "have", "had",
            "will", "would", "can", "could", "should",
            "according to", "studies show", "research indicates",
            "reported", "announced", "confirmed", "stated"
        ]
        
        claim_lower = claim.lower()
        return any(indicator in claim_lower for indicator in factual_indicators)


def extract_claims_from_text(text: str) -> List[str]:
    """
    Convenience function to extract claims from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted claims
    """
    extractor = ClaimExtractor()
    return extractor.extract_claims(text)


# Example usage
if __name__ == "__main__":
    # Test the claim extractor
    test_text = """
    The COVID-19 pandemic started in 2020. 
    Scientists believe the virus originated in China.
    The vaccine was developed in record time.
    Many people think the lockdowns were too strict.
    Studies show that masks are effective in preventing transmission.
    """
    
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(test_text)
    
    print("Extracted Claims:")
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim}")
