"""
Main fact-checker system using LangGraph for workflow orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END

# Import our components
from claim_extractor import ClaimExtractor
from web_searcher import WebSearcher
from content_crawler import ContentCrawler
from summarizer import ContentSummarizer
from verdict_analyzer import VerdictAnalyzer
from language_utils import detect_language, get_language_templates
from config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

@dataclass
class FactCheckState:
    """State object for the fact-checking workflow."""
    
    # Input
    input_text: str = ""
    
    # Language detection
    detected_language: str = "en"  # Default to English
    
    # Extracted claims
    claims: List[str] = field(default_factory=list)
    
    # Search results for each claim
    search_results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Crawled content for each claim
    crawled_content: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Summaries for each claim
    summaries: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Final verdicts for each claim
    verdicts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Overall verdict for the entire text
    overall_verdict: Dict[str, Any] = field(default_factory=dict)
    
    # Workflow metadata
    workflow_metadata: Dict[str, Any] = field(default_factory=lambda: {
        'start_time': datetime.now().isoformat(),
        'processing_time': {},
        'steps_completed': []
    })
    
    # Errors encountered during processing
    errors: List[str] = field(default_factory=list)


class FactChecker:
    """Main fact-checker system with LangGraph workflow orchestration."""
    
    def __init__(self):
        """Initialize the fact-checker with all components."""
        self.claim_extractor = ClaimExtractor()
        self.web_searcher = WebSearcher()
        self.content_crawler = ContentCrawler()
        self.summarizer = ContentSummarizer()
        self.verdict_analyzer = VerdictAnalyzer()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("FactChecker initialized with LangGraph workflow")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for fact-checking."""
        
        # Create the state graph
        workflow = StateGraph(FactCheckState)
        
        # Add nodes for each step
        workflow.add_node("extract_claims", self._extract_claims_node)
        workflow.add_node("search_claims", self._search_claims_node)
        workflow.add_node("crawl_content", self._crawl_content_node)
        workflow.add_node("summarize_content", self._summarize_content_node)
        workflow.add_node("analyze_verdicts", self._analyze_verdicts_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        
        # Define the workflow edges
        workflow.set_entry_point("extract_claims")
        
        workflow.add_edge("extract_claims", "search_claims")
        workflow.add_edge("search_claims", "crawl_content")
        workflow.add_edge("crawl_content", "summarize_content")
        workflow.add_edge("summarize_content", "analyze_verdicts")
        workflow.add_edge("analyze_verdicts", "finalize_results")
        workflow.add_edge("finalize_results", END)
        
        return workflow.compile()
    
    async def _extract_claims_node(self, state: FactCheckState) -> FactCheckState:
        """Extract claims from input text."""
        try:
            start_time = datetime.now()
            logger.info("Extracting claims from input text")
            
            if not state.input_text:
                state.errors.append("No input text provided")
                return state
            
            # Extract claims
            claims = await self.claim_extractor.extract_claims(
                state.input_text, 
                language=state.detected_language
            )
            state.claims = claims
            
            # Update metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            state.workflow_metadata['steps_completed'].append('extract_claims')
            state.workflow_metadata['processing_time']['extract_claims'] = processing_time
            
            logger.info(f"Extracted {len(claims)} claims in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error extracting claims: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def _search_claims_node(self, state: FactCheckState) -> FactCheckState:
        """Search for information about each claim."""
        try:
            start_time = datetime.now()
            logger.info(f"Searching web for {len(state.claims)} claims")
            
            if not state.claims:
                state.errors.append("No claims to search for")
                return state
            
            # Search for each claim
            search_results = await self.web_searcher.batch_search_claims(state.claims)
            state.search_results = search_results
            
            # Filter for reliable sources
            for claim in search_results:
                filtered_results = self.web_searcher.filter_reliable_sources(search_results[claim])
                state.search_results[claim] = filtered_results
            
            # Update metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            state.workflow_metadata['steps_completed'].append('search_claims')
            state.workflow_metadata['processing_time']['search_claims'] = processing_time
            
            total_results = sum(len(results) for results in search_results.values())
            logger.info(f"Found {total_results} total search results in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error searching claims: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def _crawl_content_node(self, state: FactCheckState) -> FactCheckState:
        """Crawl content from search results."""
        try:
            start_time = datetime.now()
            logger.info("Crawling content from search results")
            
            if not state.search_results:
                state.errors.append("No search results to crawl")
                return state
            
            # Collect all unique URLs first to avoid duplicates
            all_urls = set()
            url_to_claims = {}  # Track which claims each URL relates to
            
            for claim, search_results in state.search_results.items():
                for result in search_results:
                    url = result.get('url', '')
                    if url and url not in all_urls:
                        all_urls.add(url)
                        url_to_claims[url] = [claim]
                    elif url in url_to_claims:
                        url_to_claims[url].append(claim)
            
            logger.info(f"Found {len(all_urls)} unique URLs from {sum(len(results) for results in state.search_results.values())} total search results")
            
            # Crawl unique URLs only once
            url_content = {}
            if all_urls:
                urls_list = list(all_urls)[:config.crawl4ai_max_pages]  # Limit to max pages
                logger.info(f"Crawling {len(urls_list)} unique URLs...")
                
                crawled_results = await self.content_crawler.crawl_urls(urls_list, "fact-checking")
                
                for result in crawled_results:
                    url = result.get('url', '')
                    if url:
                        url_content[url] = result
            
            # Distribute crawled content to claims
            crawled_content = {}
            for claim in state.search_results.keys():
                crawled_content[claim] = []
                
                # Find all URLs related to this claim
                for url, claims in url_to_claims.items():
                    if claim in claims and url in url_content:
                        content = url_content[url].copy()
                        content['claim'] = claim  # Add claim context
                        crawled_content[claim].append(content)
            
            state.crawled_content = crawled_content
            
            # Update metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            state.workflow_metadata['steps_completed'].append('crawl_content')
            state.workflow_metadata['processing_time']['crawl_content'] = processing_time
            
            total_crawled = sum(len(content) for content in crawled_content.values())
            logger.info(f"Crawled {len(url_content)} unique pages, distributed to {total_crawled} claim-content pairs in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error crawling content: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def _summarize_content_node(self, state: FactCheckState) -> FactCheckState:
        """Summarize crawled content for each claim."""
        try:
            start_time = datetime.now()
            logger.info("Summarizing crawled content")
            
            if not state.crawled_content:
                state.errors.append("No crawled content to summarize")
                return state
            
            # Summarize content for each claim
            summaries = {}
            
            for claim, content_list in state.crawled_content.items():
                logger.info(f"Summarizing content for claim: {claim[:50]}...")
                claim_summaries = await self.summarizer.batch_summarize(content_list, claim)
                
                # Rank summaries by quality
                ranked_summaries = self.summarizer.rank_summaries_by_quality(claim_summaries)
                summaries[claim] = ranked_summaries
            
            state.summaries = summaries
            
            # Update metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            state.workflow_metadata['steps_completed'].append('summarize_content')
            state.workflow_metadata['processing_time']['summarize_content'] = processing_time
            
            total_summaries = sum(len(summ) for summ in summaries.values())
            logger.info(f"Generated {total_summaries} summaries in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error summarizing content: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def _analyze_verdicts_node(self, state: FactCheckState) -> FactCheckState:
        """Analyze summaries to generate verdicts."""
        try:
            start_time = datetime.now()
            logger.info("Analyzing summaries to generate verdicts")
            
            if not state.summaries:
                state.errors.append("No summaries to analyze")
                return state
            
            # Analyze each claim
            verdicts = await self.verdict_analyzer.batch_analyze_claims(
                state.summaries, 
                language=state.detected_language
            )
            state.verdicts = verdicts
            
            # Update metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            state.workflow_metadata['steps_completed'].append('analyze_verdicts')
            state.workflow_metadata['processing_time']['analyze_verdicts'] = processing_time
            
            logger.info(f"Generated {len(verdicts)} verdicts in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Error analyzing verdicts: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def _finalize_results_node(self, state: FactCheckState) -> FactCheckState:
        """Finalize and format results."""
        try:
            start_time = datetime.now()
            logger.info("Finalizing fact-check results")
            
            # Generate overall verdict for the entire text
            if state.verdicts and state.claims:
                logger.info("Generating overall verdict for the entire text")
                overall_verdict = await self.verdict_analyzer.generate_overall_verdict(
                    state.input_text, 
                    state.claims, 
                    state.verdicts,
                    language=state.detected_language
                )
                state.overall_verdict = overall_verdict
                logger.info(f"Overall verdict: {overall_verdict.get('overall_verdict', 'Unknown')}")
            else:
                state.overall_verdict = {
                    'overall_verdict': 'UNVERIFIED',
                    'confidence': 0.1,
                    'justification': 'No verifiable claims found or analyzed.',
                    'claim_breakdown': {},
                    'methodology': 'No claims to analyze'
                }
            
            # Calculate total processing time
            total_time = (datetime.now() - datetime.fromisoformat(state.workflow_metadata['start_time'])).total_seconds()
            
            # Update final metadata
            state.workflow_metadata.update({
                'end_time': datetime.now().isoformat(),
                'total_processing_time': total_time,
                'total_claims': len(state.claims),
                'total_search_results': sum(len(results) for results in state.search_results.values()) if state.search_results else 0,
                'total_crawled_pages': sum(len(content) for content in state.crawled_content.values()) if state.crawled_content else 0,
                'total_summaries': sum(len(summ) for summ in state.summaries.values()) if state.summaries else 0,
                'total_verdicts': len(state.verdicts) if state.verdicts else 0,
                'has_errors': len(state.errors) > 0,
                'error_count': len(state.errors)
            })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            state.workflow_metadata['processing_time']['finalize_results'] = processing_time
            
            logger.info(f"Fact-checking completed in {total_time:.2f}s total")
            
        except Exception as e:
            error_msg = f"Error finalizing results: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def check_facts(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Main method to fact-check text.
        
        Args:
            text: Input text to fact-check
            language: Language code for the text (if not provided, will be auto-detected)
            
        Returns:
            Comprehensive fact-check results
        """
        try:
            logger.info(f"Starting fact-check for text: {text[:100]}...")
            
            # Use provided language or detect it
            if language:
                detected_language = language
                logger.info(f"Using provided language: {detected_language}")
            else:
                detected_language = detect_language(text)
                logger.info(f"Auto-detected language: {detected_language}")
            
            # Initialize state
            initial_state = FactCheckState(
                input_text=text,
                detected_language=detected_language,
                errors=[],
                workflow_metadata={
                    'start_time': datetime.now().isoformat(), 
                    'processing_time': {},
                    'steps_completed': [],
                    'language': detected_language
                }
            )
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Handle case where LangGraph returns a dict instead of FactCheckState
            if isinstance(final_state, dict):
                # Convert dict back to FactCheckState if needed
                logger.warning("Workflow returned dict instead of FactCheckState, converting...")
                final_state = FactCheckState(
                    input_text=final_state.get('input_text', text),
                    claims=final_state.get('claims', []),
                    search_results=final_state.get('search_results', {}),
                    crawled_content=final_state.get('crawled_content', {}),
                    summaries=final_state.get('summaries', {}),
                    verdicts=final_state.get('verdicts', {}),
                    overall_verdict=final_state.get('overall_verdict', {}),
                    workflow_metadata=final_state.get('workflow_metadata', {}),
                    errors=final_state.get('errors', [])
                )
            elif not isinstance(final_state, FactCheckState):
                # Handle unexpected return type
                logger.error(f"Unexpected workflow return type: {type(final_state)}")
                return {
                    'success': False,
                    'error': f'Unexpected workflow return type: {type(final_state)}',
                    'input_text': text,
                    'claims': [],
                    'results': [],
                    'metadata': {
                        'error_occurred': True,
                        'error_message': f'Unexpected workflow return type: {type(final_state)}'
                    }
                }
            
            # Ensure final_state is a proper FactCheckState object
            if not hasattr(final_state, 'verdicts'):
                logger.error(f"Invalid final state - missing verdicts attribute: {type(final_state)}")
                return {
                    'success': False,
                    'error': 'Invalid final state - missing verdicts attribute',
                    'input_text': text,
                    'claims': getattr(final_state, 'claims', []),
                    'results': [],
                    'metadata': {
                        'error_occurred': True,
                        'error_message': 'Invalid final state - missing verdicts attribute'
                    }
                }
            
            # Format and return results
            return self._format_results(final_state)
            
        except Exception as e:
            logger.error(f"Error in fact-checking workflow: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'input_text': text,
                'claims': [],
                'results': [],
                'metadata': {
                    'error_occurred': True,
                    'error_message': str(e)
                }
            }
    
    def _format_results(self, state: FactCheckState) -> Dict[str, Any]:
        """Format the final results for output."""
        # Defensive check for state type and attributes
        if not isinstance(state, FactCheckState):
            logger.error(f"Invalid state object type: {type(state)}")
            logger.error(f"State contents: {state}")
            return {
                'success': False,
                'error': f'Invalid state object type: {type(state)}',
                'input_text': getattr(state, 'input_text', '') if hasattr(state, 'input_text') else '',
                'claims': getattr(state, 'claims', []) if hasattr(state, 'claims') else [],
                'results': [],
                'metadata': {
                    'error_occurred': True,
                    'error_message': f'Invalid state object type: {type(state)}'
                }
            }
        
        if not hasattr(state, 'verdicts'):
            logger.error(f"State object missing verdicts attribute: {dir(state)}")
            return {
                'success': False,
                'error': 'State object missing verdicts attribute',
                'input_text': state.input_text,
                'claims': getattr(state, 'claims', []),
                'results': [],
                'metadata': {
                    'error_occurred': True,
                    'error_message': 'State object missing verdicts attribute'
                }
            }
        
        # Ensure all required attributes exist with defaults
        input_text = getattr(state, 'input_text', '')
        claims = getattr(state, 'claims', [])
        verdicts = getattr(state, 'verdicts', {})
        overall_verdict = getattr(state, 'overall_verdict', {})
        search_results = getattr(state, 'search_results', {})
        summaries = getattr(state, 'summaries', {})
        workflow_metadata = getattr(state, 'workflow_metadata', {})
        errors = getattr(state, 'errors', [])
        
        # Compile claim results
        claim_results = []
        
        if verdicts and claims:
            for claim in claims:
                verdict = verdicts.get(claim, {})
                claim_summaries = summaries.get(claim, [])
                claim_search_results = search_results.get(claim, [])
                
                claim_result = {
                    'claim': claim,
                    'verdict': verdict.get('verdict', 'UNVERIFIED'),
                    'confidence': verdict.get('confidence', 0.0),
                    'justification': verdict.get('justification', 'No justification available'),
                    'supporting_sources': verdict.get('supporting_sources', []),
                    'contradicting_sources': verdict.get('contradicting_sources', []),
                    'key_evidence': verdict.get('key_evidence', []),
                    'limitations': verdict.get('limitations', 'No limitations noted'),
                    'quality_assessment': verdict.get('quality_assessment', {}),
                    'source_details': verdict.get('source_details', []),
                    'summary_count': len(claim_summaries),
                    'search_result_count': len(claim_search_results)
                }
                
                claim_results.append(claim_result)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(claim_results, state)
        
        # Determine success status
        success = len(errors) == 0 and len(claims) > 0
        
        return {
            'success': success,
            'input_text': input_text,
            'claims': claims,
            'overall_verdict': overall_verdict,
            'individual_claim_results': claim_results,
            'results': claim_results,  # Compatibility alias
            'overall_statistics': overall_stats,
            'metadata': workflow_metadata,
            'errors': errors if errors else None
        }
    
    def _calculate_overall_stats(self, claim_results: List[Dict], state: FactCheckState) -> Dict[str, Any]:
        """Calculate overall statistics for the fact-check session."""
        if not claim_results:
            return {
                'total_claims': 0,
                'verdicts_summary': {},
                'avg_confidence': 0.0,
                'high_confidence_claims': 0,
                'quality_distribution': {},
                'processing_summary': {
                    'total_time': 0,
                    'steps_completed': 0,
                    'total_sources_found': 0,
                    'total_pages_crawled': 0,
                    'total_summaries_generated': 0
                }
            }
        
        # Count verdicts
        verdicts_count = {}
        confidences = []
        quality_levels = []
        
        for result in claim_results:
            verdict = result.get('verdict', 'UNKNOWN')
            verdicts_count[verdict] = verdicts_count.get(verdict, 0) + 1
            
            confidence = result.get('confidence', 0)
            confidences.append(confidence)
            
            quality = result.get('quality_assessment', {}).get('quality_level', 'UNKNOWN')
            quality_levels.append(quality)
        
        # Calculate statistics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        high_confidence = sum(1 for c in confidences if c >= config.confidence_threshold)
        
        quality_distribution = {}
        for quality in quality_levels:
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        # Get metadata safely
        workflow_metadata = getattr(state, 'workflow_metadata', {})
        
        # Calculate overall verdict based on individual claim verdicts
        overall_verdict_result = self._calculate_overall_verdict(claim_results, verdicts_count, avg_confidence)
        
        return {
            'total_claims': len(claim_results),
            'verdicts_summary': verdicts_count,
            'avg_confidence': avg_confidence,
            'high_confidence_claims': high_confidence,
            'quality_distribution': quality_distribution,
            # Add overall verdict fields
            'overall_verdict': overall_verdict_result['verdict'],
            'overall_confidence': overall_verdict_result['confidence'],
            'overall_reasoning': overall_verdict_result['reasoning'],
            'processing_summary': {
                'total_time': workflow_metadata.get('total_processing_time', 0),
                'steps_completed': len(workflow_metadata.get('steps_completed', [])),
                'total_sources_found': workflow_metadata.get('total_search_results', 0),
                'total_pages_crawled': workflow_metadata.get('total_crawled_pages', 0),
                'total_summaries_generated': workflow_metadata.get('total_summaries', 0)
            }
        }
    
    def _calculate_overall_verdict(self, claim_results: List[Dict], verdicts_count: Dict, avg_confidence: float) -> Dict[str, Any]:
        """Calculate overall verdict based on individual claim verdicts."""
        if not claim_results:
            return {
                'verdict': 'INSUFFICIENT_INFO',
                'confidence': 0.0,
                'reasoning': 'No claims were extracted or analyzed'
            }
        
        total_claims = len(claim_results)
        
        # Count verdicts
        true_count = verdicts_count.get('TRUE', 0)
        false_count = verdicts_count.get('FALSE', 0) 
        partly_true_count = verdicts_count.get('PARTLY_TRUE', 0)
        unverified_count = verdicts_count.get('UNVERIFIED', 0)
        
        # Calculate percentages
        verified_claims = true_count + false_count + partly_true_count
        unverified_rate = unverified_count / total_claims
        false_rate = false_count / total_claims if total_claims > 0 else 0
        true_rate = true_count / total_claims if total_claims > 0 else 0
        
        # Decision logic
        overall_verdict = 'INSUFFICIENT_INFO'
        overall_confidence = avg_confidence
        reasoning = ''
        
        # If too many claims are unverified, insufficient info
        if unverified_rate > 0.6:
            overall_verdict = 'INSUFFICIENT_INFO'
            reasoning = f'Too many claims could not be verified ({unverified_count}/{total_claims} unverified)'
            overall_confidence = 0.0
        
        # If majority are false, not supported
        elif false_rate > 0.5:
            overall_verdict = 'NOT_SUPPORTED' 
            reasoning = f'Majority of claims are false ({false_count}/{total_claims} false)'
            overall_confidence = min(0.9, avg_confidence + 0.1)
        
        # If all true, supported
        elif true_count == total_claims:
            overall_verdict = 'SUPPORTED'
            reasoning = f'All claims are true ({true_count}/{total_claims} true)'
            overall_confidence = min(0.9, avg_confidence + 0.1)
        
        # If mostly true with some partly true, supported
        elif true_rate >= 0.7 and partly_true_count > 0:
            overall_verdict = 'SUPPORTED'
            reasoning = f'Most claims are true or partly true ({true_count + partly_true_count}/{total_claims})'
            overall_confidence = avg_confidence
        
        # If mix of true and false, mixed
        elif true_count > 0 and false_count > 0:
            overall_verdict = 'MIXED'
            reasoning = f'Mixed results: {true_count} true, {false_count} false, {partly_true_count} partly true'
            overall_confidence = avg_confidence
        
        # If mostly partly true, mixed  
        elif partly_true_count > total_claims * 0.5:
            overall_verdict = 'MIXED'
            reasoning = f'Most claims are partly true ({partly_true_count}/{total_claims})'
            overall_confidence = avg_confidence
        
        # Default case
        else:
            overall_verdict = 'INSUFFICIENT_INFO'
            reasoning = f'Unable to determine clear verdict from {total_claims} claims'
            overall_confidence = avg_confidence * 0.5
        
        return {
            'verdict': overall_verdict,
            'confidence': max(0.0, min(1.0, overall_confidence)),  # Clamp to 0-1
            'reasoning': reasoning
        }

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow structure."""
        return {
            'workflow_steps': [
                'extract_claims',
                'search_claims', 
                'crawl_content',
                'summarize_content',
                'analyze_verdicts',
                'finalize_results'
            ],
            'components': {
                'claim_extractor': 'Extract factual claims from text',
                'web_searcher': 'Search web using SearXNG',
                'content_crawler': 'Crawl web pages with Crawl4AI',
                'summarizer': 'Summarize content with AI',
                'verdict_analyzer': 'Generate verdicts and justifications'
            },
            'configuration': {
                'model': config.model_name,
                'max_claims': config.max_claims_per_text,
                'max_pages_per_claim': config.crawl4ai_max_pages,
                'search_engines': config.search_engines,
                'confidence_threshold': config.confidence_threshold
            }
        }


# Convenience function for simple usage
async def fact_check_text(text: str, language: str = None) -> Dict[str, Any]:
    """
    Simple function to fact-check text.
    
    Args:
        text: Text to fact-check
        language: Language code for the text (if not provided, will be auto-detected)
        
    Returns:
        Fact-check results
    """
    checker = FactChecker()
    return await checker.check_facts(text, language=language)


# Example usage and testing
if __name__ == "__main__":
    async def test_fact_checker():
        # Test text with claims
        test_text = """
        The COVID-19 pandemic started in 2020 and affected millions worldwide.
        Scientists developed vaccines in record time, with some showing over 90% efficacy.
        The World Health Organization declared it a pandemic in March 2020.
        Many countries implemented lockdown measures to slow the spread.
        """
        
        # Initialize fact checker
        checker = FactChecker()
        
        # Get workflow info
        workflow_info = checker.get_workflow_info()
        print("Workflow Info:")
        print(f"Steps: {workflow_info['workflow_steps']}")
        print(f"Model: {workflow_info['configuration']['model']}")
        print()
        
        # Run fact check
        print("Starting fact-check...")
        results = await checker.check_facts(test_text)
        
        # Display results
        print(f"\nFact-Check Results:")
        print(f"Success: {results['success']}")
        print(f"Total Claims: {results['overall_statistics']['total_claims']}")
        print(f"Processing Time: {results['metadata'].get('total_processing_time', 0):.2f}s")
        
        for i, result in enumerate(results['results'], 1):
            print(f"\nClaim {i}: {result['claim']}")
            print(f"Verdict: {result['verdict']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Quality: {result['quality_assessment'].get('quality_level', 'Unknown')}")
            print(f"Justification: {result['justification'][:100]}...")
    
    # Run the test
    asyncio.run(test_fact_checker())
