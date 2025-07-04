"""
Content crawler component using Crawl4AI for the fact-checker system.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from crawl4ai import AsyncWebCrawler, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from config import config

logger = logging.getLogger(__name__)

class ContentCrawler:
    """Web content crawler using Crawl4AI."""
    
    def __init__(self):
        """Initialize the content crawler."""
        self.max_pages = config.crawl4ai_max_pages
        self.timeout = config.crawl4ai_timeout
        self.max_content_length = config.crawl4ai_max_content_length
        logger.info("ContentCrawler initialized")
    
    async def crawl_urls(self, urls: List[str], claim: str) -> List[Dict[str, Any]]:
        """
        Crawl multiple URLs and extract relevant content.
        
        Args:
            urls: List of URLs to crawl
            claim: The claim being fact-checked (for context)
            
        Returns:
            List of crawled content with metadata
        """
        crawled_content = []
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            # Limit to max pages
            urls_to_crawl = urls[:self.max_pages]
            
            logger.info(f"Crawling {len(urls_to_crawl)} URLs for claim: {claim[:50]}...")
            
            for i, url in enumerate(urls_to_crawl):
                try:
                    content = await self._crawl_single_url(crawler, url, claim)
                    if content:
                        content['crawl_order'] = i + 1
                        crawled_content.append(content)
                        
                except Exception as e:
                    logger.error(f"Error crawling URL {url}: {e}")
                    continue
        
        logger.info(f"Successfully crawled {len(crawled_content)} pages")
        return crawled_content
    
    async def _crawl_single_url(self, crawler: AsyncWebCrawler, url: str, claim: str) -> Optional[Dict[str, Any]]:
        """
        Crawl a single URL and extract relevant content.
        
        Args:
            crawler: AsyncWebCrawler instance
            url: URL to crawl
            claim: The claim for context
              Returns:
            Extracted content with metadata
        """
        try:
            logger.info(f"Crawling: {url}")
            
            # Crawl the page without LLM extraction for now to ensure basic functionality
            result = await crawler.arun(
                url=url,
                bypass_cache=True,
                timeout=self.timeout
            )
            
            if not result.success:
                logger.warning(f"Failed to crawl {url}: {result.error_message}")
                return None
            
            # Extract and clean content
            content = self._process_crawled_content(result, url, claim)
            
            return content
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return None
    
    def _create_extraction_strategy(self, claim: str) -> LLMExtractionStrategy:
        """Create an extraction strategy focused on the claim."""
        extraction_prompt = f"""
        Extract information relevant to fact-checking this claim: "{claim}"
        
        Focus on:
        1. Facts, statistics, and data related to the claim
        2. Expert quotes and authoritative statements
        3. Source citations and references
        4. Dates and temporal context
        5. Contradictory information if present
        
        Ignore:
        1. Navigation menus and advertisements
        2. Unrelated content
        3. Comments sections
        4. Social media widgets
        
        Extract the content in a structured format with:
        - Main facts relevant to the claim
        - Supporting evidence or data
        - Expert opinions or authoritative sources
        - Any contradictory information        - Publication date and source information
        """
        
        return LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="google/gemini-1.5-flash",
                api_token=config.google_ai_api_key
            ),
            instruction=extraction_prompt
        )
    
    def _process_crawled_content(self, result, url: str, claim: str) -> Dict[str, Any]:
        """Process and structure the crawled content."""
        # Get basic page information
        page_title = getattr(result, 'title', '') or 'No title'
        page_text = result.markdown or result.cleaned_html or ''
        
        # Limit content length
        if len(page_text) > self.max_content_length:
            page_text = page_text[:self.max_content_length] + "... [content truncated]"
        
        # Get the extracted content (use page_text if no LLM extraction)
        extracted_data = result.extracted_content or page_text
        
        # Structure the content
        content = {
            'url': url,
            'title': page_title,
            'extracted_content': extracted_data,
            'full_text': page_text,
            'claim': claim,
            'metadata': {
                'crawl_timestamp': result.response_headers.get('date', ''),
                'content_type': result.response_headers.get('content-type', ''),
                'content_length': len(page_text),
                'success': result.success,
                'status_code': getattr(result, 'status_code', None)
            }
        }
        
        # Extract additional metadata
        content['metadata'].update(self._extract_content_metadata(page_text, page_title))
        
        return content
    
    def _extract_content_metadata(self, text: str, title: str) -> Dict[str, Any]:
        """Extract additional metadata from the content."""
        import re
        from datetime import datetime
        
        metadata = {}
        
        # Look for publication dates
        date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # MM/DD/YYYY
            r'\b(\d{4}-\d{2}-\d{2})\b',      # YYYY-MM-DD
            r'\b(\w+ \d{1,2}, \d{4})\b',     # Month DD, YYYY
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates_found.extend(matches)
        
        metadata['dates_found'] = dates_found[:5]  # Limit to 5 dates
        
        # Look for author information
        author_patterns = [
            r'[Bb]y\s+([A-Z][a-z]+ [A-Z][a-z]+)',
            r'[Aa]uthor:\s*([A-Z][a-z]+ [A-Z][a-z]+)',
            r'[Ww]ritten by\s+([A-Z][a-z]+ [A-Z][a-z]+)'
        ]
        
        authors_found = []
        for pattern in author_patterns:
            matches = re.findall(pattern, text)
            authors_found.extend(matches)
        
        metadata['authors_found'] = list(set(authors_found))[:3]  # Limit and dedupe
        
        # Count key elements
        metadata['word_count'] = len(text.split())
        metadata['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Look for credibility indicators
        credibility_indicators = [
            'peer-reviewed', 'study', 'research', 'according to',
            'expert', 'professor', 'dr.', 'phd', 'university',
            'published', 'journal', 'clinical trial'
        ]
        
        found_indicators = []
        text_lower = text.lower()
        for indicator in credibility_indicators:
            if indicator in text_lower:
                found_indicators.append(indicator)
        
        metadata['credibility_indicators'] = found_indicators
        
        return metadata
    
    async def crawl_search_results(self, search_results: List[Dict[str, Any]], claim: str) -> List[Dict[str, Any]]:
        """
        Crawl URLs from search results.
        
        Args:
            search_results: List of search results with URLs
            claim: The claim being fact-checked
            
        Returns:
            List of crawled content
        """
        # Extract URLs from search results
        urls = []
        for result in search_results:
            url = result.get('url')
            if url and self._is_crawlable_url(url):
                urls.append(url)
        
        # Crawl the URLs
        return await self.crawl_urls(urls, claim)
    
    def _is_crawlable_url(self, url: str) -> bool:
        """Check if a URL is suitable for crawling."""
        # Basic URL validation
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        # Skip certain file types
        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip certain domains that are hard to crawl
        skip_domains = ['youtube.com', 'facebook.com', 'twitter.com', 'instagram.com']
        if any(domain in url.lower() for domain in skip_domains):
            return False
        
        return True
    
    def summarize_crawled_content(self, crawled_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of all crawled content.
        
        Args:
            crawled_content: List of crawled content
            
        Returns:
            Summary statistics and overview
        """
        if not crawled_content:
            return {
                'total_pages': 0,
                'total_content_length': 0,
                'successful_crawls': 0,
                'avg_content_length': 0,
                'domains_crawled': []
            }
        
        successful_crawls = len(crawled_content)
        total_length = sum(len(content.get('full_text', '')) for content in crawled_content)
        avg_length = total_length / successful_crawls if successful_crawls > 0 else 0
        
        # Extract domains
        from urllib.parse import urlparse
        domains = []
        for content in crawled_content:
            url = content.get('url', '')
            if url:
                try:
                    domain = urlparse(url).netloc
                    domains.append(domain)
                except:
                    continue
        
        return {
            'total_pages': successful_crawls,
            'total_content_length': total_length,
            'successful_crawls': successful_crawls,
            'avg_content_length': avg_length,
            'domains_crawled': list(set(domains)),
            'content_sources': [
                {
                    'url': content.get('url', ''),
                    'title': content.get('title', ''),
                    'length': len(content.get('full_text', ''))
                }
                for content in crawled_content
            ]
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_crawler():
        crawler = ContentCrawler()
        
        # Test URLs
        urls = [
            "https://www.reuters.com",
            "https://www.bbc.com/news"
        ]
        
        claim = "COVID-19 vaccines are effective"
        
        results = await crawler.crawl_urls(urls, claim)
        
        print(f"Crawled {len(results)} pages")
        for result in results:
            print(f"- {result['title'][:50]}... ({len(result['full_text'])} chars)")
    
    # Run test
    asyncio.run(test_crawler())
