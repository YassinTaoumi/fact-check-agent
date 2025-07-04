"""
Content summarization component for the fact-checker system.
"""

import logging
import asyncio
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from config import config, SUMMARIZATION_PROMPT
from rate_limiter import llm_rate_limiter

logger = logging.getLogger(__name__)

class ContentSummarizer:
    """Summarize crawled content for fact-checking analysis."""
    
    def __init__(self):
        """Initialize the content summarizer."""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=config.google_ai_api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        self.max_summary_length = config.max_summary_length
        logger.info("ContentSummarizer initialized")
    
    async def summarize_content(self, content: Dict[str, Any], claim: str) -> Dict[str, Any]:
        """
        Summarize a single piece of content for a specific claim.
        
        Args:
            content: Crawled content with text and metadata
            claim: The claim being fact-checked
            
        Returns:
            Summary with analysis
        """
        try:
            # Extract the text content
            text_content = self._extract_text_for_summary(content)
            
            if not text_content or len(text_content.strip()) < config.min_content_length:
                # If content is too short but still exists, try to use it anyway
                if text_content and len(text_content.strip()) >= 20:
                    logger.info(f"Using short content for summarization: {content.get('url', 'unknown')}")
                    # Continue with processing even for short content
                else:
                    logger.warning(f"Content too short for summarization: {content.get('url', 'unknown')}")
                    return self._create_empty_summary(content, "Content too short")
            
            # Generate summary using AI
            summary = await self._generate_ai_summary(text_content, claim)
            
            # Create structured summary result
            summary_result = {
                'source_url': content.get('url', ''),
                'source_title': content.get('title', ''),
                'claim': claim,
                'summary': summary,
                'original_length': len(text_content),
                'summary_length': len(summary),
                'metadata': content.get('metadata', {}),
                'credibility_score': self._assess_source_credibility(content),
                'relevance_score': self._assess_content_relevance(text_content, claim),
                'summary_timestamp': self._get_current_timestamp()
            }
            
            logger.info(f"Summarized content from {content.get('url', 'unknown')}")
            return summary_result
            
        except Exception as e:
            logger.error(f"Error summarizing content: {e}")
            return self._create_empty_summary(content, str(e))
    
    async def _generate_ai_summary(self, content: str, claim: str) -> str:
        """Generate AI summary of content for the claim."""
        try:
            # Prepare the prompt
            prompt = SUMMARIZATION_PROMPT.format(
                claim=claim,
                content=content,
                max_length=self.max_summary_length
            )
            
            # Get AI response with rate limiting
            messages = [
                SystemMessage(content="You are a professional fact-checking assistant focused on accurate summarization."),
                HumanMessage(content=prompt)
            ]
            
            await llm_rate_limiter.wait_if_needed()
            response = self.llm(messages)
            summary = response.content.strip()
            
            # Ensure summary isn't too long
            if len(summary) > self.max_summary_length * 5:  # Character limit (roughly 5x word limit)
                summary = summary[:self.max_summary_length * 5] + "... [summary truncated]"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def _extract_text_for_summary(self, content: Dict[str, Any]) -> str:
        """Extract the best text content for summarization."""
        # Priority order for text content
        text_sources = [
            content.get('extracted_content', ''),  # AI-extracted content
            content.get('full_text', ''),          # Full markdown/text
            content.get('title', '')               # Fallback to title
        ]
        
        for text in text_sources:
            if text and len(text.strip()) >= config.min_content_length:
                return text.strip()
        
        return ""
    
    def _assess_source_credibility(self, content: Dict[str, Any]) -> float:
        """Assess the credibility of the content source."""
        url = content.get('url', '').lower()
        metadata = content.get('metadata', {})
        
        credibility_score = 0.5  # Base score
        
        # High credibility domains
        high_credibility = [
            'reuters.com', 'apnews.com', 'bbc.com', 'npr.org',
            'snopes.com', 'factcheck.org', 'politifact.com',
            'nature.com', 'sciencemag.org', 'pubmed.ncbi.nlm.nih.gov'
        ]
        
        # Government and educational domains
        gov_edu_domains = ['.gov', '.edu', 'who.int', 'cdc.gov', 'nih.gov']
        
        # Check domain credibility
        for domain in high_credibility:
            if domain in url:
                credibility_score = 0.9
                break
        
        for domain in gov_edu_domains:
            if domain in url:
                credibility_score = max(credibility_score, 0.8)
                break
        
        # Check for credibility indicators in metadata
        indicators = metadata.get('credibility_indicators', [])
        if indicators:
            credibility_score += min(len(indicators) * 0.05, 0.2)  # Up to 0.2 bonus
        
        # Check for author information
        authors = metadata.get('authors_found', [])
        if authors:
            credibility_score += 0.1
        
        # Ensure score is between 0 and 1
        return min(max(credibility_score, 0.0), 1.0)
    
    def _assess_content_relevance(self, content: str, claim: str) -> float:
        """Assess how relevant the content is to the claim with improved accuracy."""
        if not content or not claim:
            return 0.0
        
        content_lower = content.lower()
        claim_lower = claim.lower()
        
        # Extract key terms from the claim (remove stop words)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'for', 'with', 'his', 'they', 'be', 'he', 'have', 'it', 'that', 'was', 'i', 'you', 'this', 'but', 'will', 'or', 'from', 'has', 'by', 'not', 'can', 'had', 'an', 'all', 'if'}
        claim_words = set(word.strip('.,!?;:()[]') for word in claim_lower.split())
        claim_words = {word for word in claim_words if len(word) > 2 and word not in stop_words}
        
        if not claim_words:
            return 0.0
        
        # Count exact word matches
        exact_matches = sum(1 for word in claim_words if word in content_lower)
        word_relevance = exact_matches / len(claim_words)
        
        # Extract and match key entities and phrases
        claim_entities = self._extract_entities(claim_lower)
        content_entities = self._extract_entities(content_lower)
        
        entity_matches = 0
        for entity in claim_entities:
            if entity in content_entities or entity in content_lower:
                entity_matches += 1
        
        entity_relevance = entity_matches / len(claim_entities) if claim_entities else 0
        
        # Look for phrase matches (2-3 word sequences)
        claim_phrases = self._extract_key_phrases(claim_lower)
        phrase_matches = sum(1 for phrase in claim_phrases if phrase in content_lower)
        phrase_relevance = phrase_matches / len(claim_phrases) if claim_phrases else 0
        
        # Look for semantic indicators
        semantic_score = self._assess_semantic_relevance(content_lower, claim_lower)
        
        # Combine scores with weights
        relevance_score = (
            word_relevance * 0.3 +
            entity_relevance * 0.3 + 
            phrase_relevance * 0.25 +
            semantic_score * 0.15
        )
        
        # Boost score for direct claims or contradictions
        if any(indicator in content_lower for indicator in ['false', 'true', 'confirmed', 'denied', 'debunked', 'verified']):
            relevance_score += 0.2
        
        # Ensure score is between 0 and 1
        return min(relevance_score, 1.0)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities (names, places, organizations) from text."""
        entities = []
        words = text.split()
        
        # Look for capitalized words/phrases (potential proper nouns)
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 2:
                # Check if it's part of a multi-word entity
                entity = word
                j = i + 1
                while j < len(words) and j < i + 3:  # Max 3-word entities
                    if words[j] and words[j][0].isupper():
                        entity += " " + words[j]
                        j += 1
                    else:
                        break
                entities.append(entity.lower())
        
        return entities
    
    def _assess_semantic_relevance(self, content: str, claim: str) -> float:
        """Assess semantic relevance using keyword patterns."""
        # Look for semantic patterns that indicate relevance
        relevance_patterns = [
            'according to', 'reports show', 'evidence suggests', 'studies indicate',
            'officials say', 'confirmed that', 'denied that', 'announced that',
            'investigation found', 'analysis shows', 'data reveals'
        ]
        
        semantic_matches = sum(1 for pattern in relevance_patterns if pattern in content)
        return min(semantic_matches * 0.2, 1.0)
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """Extract key phrases from text (simple implementation)."""
        words = text.split()
        phrases = []
        
        # Extract 2-3 word phrases
        for i in range(len(words) - 1):
            phrase = ' '.join(words[i:i+2])
            if len(phrase) > 5:  # Skip very short phrases
                phrases.append(phrase)
        
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if len(phrase) > 8:  # Skip short 3-word phrases
                phrases.append(phrase)
        
        return phrases[:max_phrases]
    
    def _create_empty_summary(self, content: Dict[str, Any], error_reason: str) -> Dict[str, Any]:
        """Create an empty summary result for failed cases."""
        return {
            'source_url': content.get('url', ''),
            'source_title': content.get('title', ''),
            'claim': '',
            'summary': '',
            'original_length': 0,
            'summary_length': 0,
            'metadata': content.get('metadata', {}),
            'credibility_score': 0.0,
            'relevance_score': 0.0,
            'summary_timestamp': self._get_current_timestamp(),
            'error': error_reason
        }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def batch_summarize(self, contents: List[Dict[str, Any]], claim: str) -> List[Dict[str, Any]]:
        """
        Summarize multiple pieces of content for a claim.
        
        Args:
            contents: List of crawled content
            claim: The claim being fact-checked
            
        Returns:
            List of summaries
        """
        summaries = []
        
        logger.info(f"Summarizing {len(contents)} pieces of content for claim")
        
        for i, content in enumerate(contents):
            logger.info(f"Summarizing content {i+1}/{len(contents)}")
            summary = await self.summarize_content(content, claim)
            summaries.append(summary)
        
        return summaries
    
    def rank_summaries_by_quality(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank summaries by quality (credibility + relevance).
        
        Args:
            summaries: List of summary dictionaries
            
        Returns:
            Sorted list of summaries
        """
        def quality_score(summary):
            credibility = summary.get('credibility_score', 0)
            relevance = summary.get('relevance_score', 0)
            # Weight credibility more heavily
            return (credibility * 0.7) + (relevance * 0.3)
        
        return sorted(summaries, key=quality_score, reverse=True)
    
    def get_summary_statistics(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics about the summaries.
        
        Args:
            summaries: List of summaries
            
        Returns:
            Summary statistics
        """
        if not summaries:
            return {
                'total_summaries': 0,
                'avg_credibility': 0,
                'avg_relevance': 0,
                'total_original_length': 0,
                'total_summary_length': 0
            }
        
        valid_summaries = [s for s in summaries if not s.get('error')]
        
        if not valid_summaries:
            return {
                'total_summaries': len(summaries),
                'valid_summaries': 0,
                'avg_credibility': 0,
                'avg_relevance': 0,
                'total_original_length': 0,
                'total_summary_length': 0,
                'errors': len(summaries)
            }
        
        credibility_scores = [s.get('credibility_score', 0) for s in valid_summaries]
        relevance_scores = [s.get('relevance_score', 0) for s in valid_summaries]
        original_lengths = [s.get('original_length', 0) for s in valid_summaries]
        summary_lengths = [s.get('summary_length', 0) for s in valid_summaries]
        
        return {
            'total_summaries': len(summaries),
            'valid_summaries': len(valid_summaries),
            'avg_credibility': sum(credibility_scores) / len(credibility_scores),
            'avg_relevance': sum(relevance_scores) / len(relevance_scores),
            'total_original_length': sum(original_lengths),
            'total_summary_length': sum(summary_lengths),
            'compression_ratio': sum(summary_lengths) / sum(original_lengths) if sum(original_lengths) > 0 else 0,
            'errors': len(summaries) - len(valid_summaries)
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_summarizer():
        summarizer = ContentSummarizer()
        
        # Test content
        test_content = {
            'url': 'https://example.com/test',
            'title': 'Test Article About COVID-19 Vaccines',
            'full_text': '''
            COVID-19 vaccines have been shown to be highly effective in preventing severe illness.
            Clinical trials demonstrated efficacy rates of over 90% for several vaccines.
            The vaccines work by training the immune system to recognize the virus.
            Side effects are generally mild and temporary.
            Millions of people have been safely vaccinated worldwide.
            ''',
            'metadata': {
                'credibility_indicators': ['clinical trials', 'study'],
                'authors_found': ['Dr. Smith']
            }
        }
        
        claim = "COVID-19 vaccines are highly effective"
        
        summary = await summarizer.summarize_content(test_content, claim)
        
        print(f"Summary for claim: {claim}")
        print(f"Source: {summary['source_url']}")
        print(f"Credibility: {summary['credibility_score']:.2f}")
        print(f"Relevance: {summary['relevance_score']:.2f}")
        print(f"Summary: {summary['summary']}")
    
    # Run test
    asyncio.run(test_summarizer())
