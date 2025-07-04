"""
Web search component using SearXNG for the fact-checker system.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, quote
from config import config

logger = logging.getLogger(__name__)

class WebSearcher:
    """Web search functionality using SearXNG."""
    
    def __init__(self):
        """Initialize the web searcher."""
        self.searxng_url = config.searxng_url
        self.timeout = config.searxng_timeout
        self.max_results = config.search_results_per_claim
        self.engines = config.search_engines
        logger.info(f"WebSearcher initialized with SearXNG at {self.searxng_url}")
    
    async def search_claim(self, claim: str, session: Optional[aiohttp.ClientSession] = None) -> List[Dict[str, Any]]:
        """
        Search for information about a specific claim.
        
        Args:
            claim: The claim to search for
            session: Optional aiohttp session
            
        Returns:
            List of search results
        """
        try:
            # Create session if not provided
            if session is None:
                async with aiohttp.ClientSession() as session:
                    return await self._perform_search(claim, session)
            else:
                return await self._perform_search(claim, session)
                
        except Exception as e:
            logger.error(f"Error searching for claim '{claim}': {e}")
            return []
    
    async def _perform_search(self, claim: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Perform the actual search request."""
        try:
            # Prepare search query
            query = self._prepare_search_query(claim)
            
            # Build SearXNG search URL
            search_url = self._build_search_url(query)
            
            logger.info(f"Searching for: '{query}'")
            
            # Perform search request
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with session.get(search_url, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    results = self._parse_search_results(data)
                    logger.info(f"Found {len(results)} results for claim")
                    return results
                else:
                    logger.error(f"Search request failed with status {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error(f"Search request timed out for claim: {claim}")
            return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _prepare_search_query(self, claim: str) -> str:
        """Prepare and optimize the search query."""
        # Clean the claim
        query = claim.strip()
        
        # Remove quotes and unnecessary punctuation
        query = query.replace('"', '').replace("'", "")
        
        # Add fact-checking keywords to improve relevance
        fact_check_terms = ["fact check", "verify", "true false", "evidence"]
        
        # For very long claims, extract key terms
        if len(query) > 100:
            # Simple keyword extraction (could be improved with NLP)
            words = query.split()
            # Keep important words (nouns, proper nouns, numbers, dates)
            important_words = []
            for word in words:
                if (word[0].isupper() or  # Proper nouns
                    word.isdigit() or     # Numbers
                    len(word) > 6):       # Longer words tend to be more specific
                    important_words.append(word)
            
            if important_words:
                query = " ".join(important_words[:10])  # Limit to 10 key terms
        
        return query
    
    def _build_search_url(self, query: str) -> str:
        """Build the SearXNG search URL."""
        # Encode the query
        encoded_query = quote(query)
        
        # Build parameters
        params = {
            'q': encoded_query,
            'format': 'json',
            'engines': ','.join(self.engines),
            'categories': 'general',
            'time_range': '',  # All time
            'pageno': 1,
            'safesearch': 0
        }
        
        # Build URL
        param_string = '&'.join(f"{k}={v}" for k, v in params.items())
        search_url = urljoin(self.searxng_url, f"/search?{param_string}")
        
        return search_url
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse SearXNG search results."""
        results = []
        
        search_results = data.get('results', [])
        
        for result in search_results[:self.max_results]:
            parsed_result = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'engine': result.get('engine', ''),
                'score': result.get('score', 0),
                'publishedDate': result.get('publishedDate', ''),
                'category': result.get('category', '')
            }
            
            # Filter out invalid results
            if parsed_result['url'] and parsed_result['title']:
                results.append(parsed_result)
        
        return results
    
    async def batch_search_claims(self, claims: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for multiple claims concurrently.
        
        Args:
            claims: List of claims to search
            
        Returns:
            Dictionary mapping claims to search results
        """
        results = {}
        
        async with aiohttp.ClientSession() as session:
            # Create search tasks
            tasks = []
            for claim in claims:
                task = self.search_claim(claim, session)
                tasks.append((claim, task))
            
            # Execute searches concurrently
            logger.info(f"Searching for {len(claims)} claims concurrently")
            
            for claim, task in tasks:
                try:
                    search_results = await task
                    results[claim] = search_results
                except Exception as e:
                    logger.error(f"Error searching claim '{claim}': {e}")
                    results[claim] = []
        
        return results
    
    def filter_reliable_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter search results to prioritize reliable sources.
        
        Args:
            results: List of search results
            
        Returns:
            Filtered and ranked results
        """
        # Define reliable source domains (can be extended)
        reliable_domains = {
            # News organizations
            'reuters.com': 10,
            'apnews.com': 10,
            'bbc.com': 9,
            'cnn.com': 8,
            'npr.org': 9,
            'theguardian.com': 8,
            'nytimes.com': 8,
            'washingtonpost.com': 8,
            
            # Fact-checking sites
            'snopes.com': 10,
            'factcheck.org': 10,
            'politifact.com': 10,
            'factchecker.org': 10,
            
            # Academic and scientific
            'nature.com': 10,
            'sciencemag.org': 10,
            'pubmed.ncbi.nlm.nih.gov': 10,
            'who.int': 9,
            'cdc.gov': 9,
            
            # Government sources
            '.gov': 8,
            '.edu': 7,
            '.org': 6
        }
        
        scored_results = []
        
        for result in results:
            url = result.get('url', '').lower()
            reliability_score = 1  # Default score
            
            # Check for reliable domains
            for domain, score in reliable_domains.items():
                if domain in url:
                    reliability_score = score
                    break
            
            # Add reliability score to result
            result['reliability_score'] = reliability_score
            scored_results.append(result)
        
        # Sort by reliability score and original score
        scored_results.sort(
            key=lambda x: (x['reliability_score'], x.get('score', 0)), 
            reverse=True
        )
        
        return scored_results
    
    def get_search_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of search results.
        
        Args:
            results: List of search results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {
                'total_results': 0,
                'engines_used': [],
                'avg_reliability': 0,
                'has_fact_check_sites': False
            }
        
        engines = list(set(result.get('engine', '') for result in results))
        reliabilities = [result.get('reliability_score', 1) for result in results]
        
        fact_check_domains = ['snopes.com', 'factcheck.org', 'politifact.com']
        has_fact_check = any(
            any(domain in result.get('url', '') for domain in fact_check_domains)
            for result in results
        )
        
        return {
            'total_results': len(results),
            'engines_used': engines,
            'avg_reliability': sum(reliabilities) / len(reliabilities) if reliabilities else 0,
            'has_fact_check_sites': has_fact_check,
            'top_domains': self._get_top_domains(results)
        }
    
    def _get_top_domains(self, results: List[Dict[str, Any]]) -> List[str]:
        """Get the top domains from search results."""
        from urllib.parse import urlparse
        
        domains = []
        for result in results:
            url = result.get('url', '')
            if url:
                try:
                    domain = urlparse(url).netloc
                    domains.append(domain)
                except:
                    continue
        
        # Count domains
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Return top 5 domains
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, count in sorted_domains[:5]]


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_search():
        searcher = WebSearcher()
        
        claim = "COVID-19 vaccines are 95% effective"
        results = await searcher.search_claim(claim)
        
        print(f"Search Results for: {claim}")
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Engine: {result['engine']}")
            print(f"   Score: {result.get('score', 'N/A')}")
    
    # Run test
    asyncio.run(test_search())
