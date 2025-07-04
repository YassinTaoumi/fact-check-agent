import json
import logging
import asyncio
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import re
import sys
import os

# Add path to parent directory for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_normalizer import normalize_result, get_text_preview

try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logging.warning("Crawl4AI not available. Please install with: pip install crawl4ai")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkCrawler:
    """
    Link crawler that processes database records with content_type='link'
    and extracts text content using Crawl4AI
    """
    
    def __init__(self):
        self.crawler = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        if CRAWL4AI_AVAILABLE:
            self.crawler = AsyncWebCrawler()
            await self.crawler.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    def is_valid_url(self, url: str) -> bool:
        """Check if the provided string is a valid URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def extract_urls_from_content(self, content: str) -> list:
        """Extract URLs from content using regex"""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.findall(content)
    
    async def crawl_url(self, url: str) -> Dict[str, Any]:
        """
        Crawl a single URL and extract text content
        
        Args:
            url: The URL to crawl
            
        Returns:
            Dict containing crawled data or error information
        """
        if not CRAWL4AI_AVAILABLE:
            return {
                "success": False,
                "error": "Crawl4AI not available",
                "url": url,
                "content": None
            }
        
        if not self.is_valid_url(url):
            return {
                "success": False,
                "error": "Invalid URL format",
                "url": url,
                "content": None            }
        
        try:
            logger.info(f"Crawling URL: {url}")
            
            # Crawl the URL
            result = await self.crawler.arun(
                url=url,
                word_count_threshold=10,  # Minimum words to extract
                only_text=True,  # Extract only text content
                bypass_cache=True  # Always get fresh content
            )
            
            if result.success:
                # Extract the main content
                content = result.markdown if hasattr(result, 'markdown') else result.text
                  # Clean up the content
                content = self.clean_extracted_content(content)
                
                logger.info(f"Successfully crawled {url} - {len(content)} characters extracted")
                
                # Prepare result for normalization
                result_data = {
                    "success": True,
                    "url": url,
                    "extracted_text": content,
                    "title": getattr(result, 'title', ''),
                    "description": getattr(result, 'description', ''),
                    "word_count": len(content.split()) if content else 0,
                    "error": None
                }
                
                # Apply text normalization
                normalized_result = normalize_result(result_data, "extracted_text")
                
                # Convert back to expected format
                return {
                    "success": True,
                    "url": url,
                    "content": normalized_result["extracted_text"],
                    "title": getattr(result, 'title', ''),
                    "description": getattr(result, 'description', ''),
                    "word_count": len(normalized_result["extracted_text"].split()) if normalized_result["extracted_text"] else 0,
                    "error": None
                }
            else:
                logger.error(f"Failed to crawl {url}: {result.error_message}")
                return {
                    "success": False,
                    "error": f"Crawling failed: {result.error_message}",
                    "url": url,
                    "content": None
                }
                
        except Exception as e:
            logger.error(f"Exception while crawling {url}: {str(e)}")
            return {
                "success": False,
                "error": f"Exception: {str(e)}",
                "url": url,
                "content": None
            }
    
    def clean_extracted_content(self, content: str) -> str:
        """Clean and normalize extracted content"""
        if not content:
            return ""
        
        # Remove excessive whitespace and newlines
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Multiple newlines to double
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces/tabs to single space
        content = content.strip()
        
        return content
    
    async def process_database_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a database record and crawl links if content_type is 'link'
        
        Args:
            record: Database record with fields like content_type, content, etc.
            
        Returns:
            Updated record with crawled content or original record if not a link
        """
        try:
            # Check if this is a link content type
            content_type = record.get('content_type', '').lower()
            
            if content_type != 'link':
                logger.info(f"Record {record.get('ID', record.get('message_id', 'unknown'))} is not a link (type: {content_type})")
                return {
                    **record,
                    "crawl_status": "skipped",
                    "crawl_reason": f"Not a link content type: {content_type}"
                }
            
            # Extract URL from content (try both 'content' and 'raw_text' fields)
            content = record.get('content') or record.get('raw_text', '')
            if not content:
                return {
                    **record,
                    "crawl_status": "failed",
                    "crawl_reason": "No content found in record"
                }
            
            # Extract URLs from content
            urls = self.extract_urls_from_content(content)
            
            if not urls:
                # Check if content itself is a URL
                if self.is_valid_url(content.strip()):
                    urls = [content.strip()]
                else:
                    return {
                        **record,
                        "crawl_status": "failed",
                        "crawl_reason": "No valid URLs found in content"
                    }
            
            # Crawl the first URL found
            url = urls[0]
            crawl_result = await self.crawl_url(url)
            
            if crawl_result["success"]:
                # Update record with crawled content
                updated_record = {
                    **record,
                    "crawled_content": crawl_result["content"],
                    "crawled_title": crawl_result.get("title", ""),
                    "crawled_description": crawl_result.get("description", ""),
                    "crawled_url": crawl_result["url"],
                    "crawled_word_count": crawl_result.get("word_count", 0),
                    "crawl_status": "success",
                    "crawl_reason": None
                }
                
                logger.info(f"Successfully processed link record {record.get('ID', record.get('message_id', 'unknown'))}")
                return updated_record
            else:
                return {
                    **record,
                    "crawl_status": "failed",
                    "crawl_reason": crawl_result["error"]
                }
                
        except Exception as e:
            logger.error(f"Error processing record {record.get('ID', record.get('message_id', 'unknown'))}: {str(e)}")
            return {
                **record,
                "crawl_status": "error",
                "crawl_reason": f"Processing error: {str(e)}"
            }
    
    async def process_records_batch(self, records: list) -> list:
        """
        Process multiple database records in batch
        
        Args:
            records: List of database records
            
        Returns:
            List of processed records
        """
        
        logger.info(f"Processing batch of {len(records)} records")
        
        processed_records = []
        
        for record in records:
            processed_record = await self.process_database_record(record)
            processed_records.append(processed_record)
        
        # Summary
        success_count = sum(1 for r in processed_records if r.get('crawl_status') == 'success')
        failed_count = sum(1 for r in processed_records if r.get('crawl_status') == 'failed')
        skipped_count = sum(1 for r in processed_records if r.get('crawl_status') == 'skipped')
        
        logger.info(f"Batch processing complete: {success_count} success, {failed_count} failed, {skipped_count} skipped")
        
        return processed_records


# Utility functions for easy usage
async def crawl_link_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to crawl a single database record
    
    Args:
        record: Database record with fields like content_type, content, etc.
        
    Returns:
        Updated record with crawled content
    """
    async with LinkCrawler() as crawler:
        return await crawler.process_database_record(record)


async def crawl_link_records(records: list) -> list:
    """
    Convenience function to crawl multiple database records
    
    Args:
        records: List of database records
        
    Returns:
        List of updated records with crawled content
    """
    async with LinkCrawler() as crawler:
        return await crawler.process_records_batch(records)


def main():
    """
    Main function for command-line usage.
    Accepts JSON input from stdin or command line argument.
    """
    import sys
    try:
        # Get input JSON
        if len(sys.argv) > 1:
            # Input provided as command line argument
            input_json = sys.argv[1]
        else:
            # Input provided via stdin
            input_json = sys.stdin.read().strip()
        
        if not input_json:
            print(json.dumps({"error": "No input provided"}))
            sys.exit(1)
        
        # Parse JSON input
        try:
            record = json.loads(input_json)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}))
            sys.exit(1)
        
        # Process the record asynchronously
        result = asyncio.run(crawl_link_record(record))
        
        # Output results as JSON
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "ID": record.get("ID") if 'record' in locals() else None
        }
        print(json.dumps(error_result))
        sys.exit(1)


async def test_main():
    """Test function for async testing"""
    
    # Example database record
    sample_record = {
        "message_id": "test_123",
        "chat_jid": "test@s.whatsapp.net",
        "content": "https://example.com/article",
        "content_type": "link",
        "sender_jid": "sender@s.whatsapp.net",
        "timestamp": "2025-06-20T00:00:00Z"
    }
    
    print("Testing Link Crawler with sample record...")
    
    # Test single record processing
    result = await crawl_link_record(sample_record)
    
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test batch processing
    records = [sample_record]
    batch_results = await crawl_link_records(records)
    
    print(f"Batch results: {len(batch_results)} records processed")


if __name__ == "__main__":
    main()
