import json
import sys
import logging
from typing import Dict, Any
import re
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLinkCrawler:
    """
    Simple link crawler for testing that doesn't require crawl4ai
    Returns mock data for URL links to test the pipeline
    """
    
    def __init__(self):
        pass
    
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
    
    def crawl_url(self, url: str) -> Dict[str, Any]:
        """
        Mock crawl a URL and return test content
        
        Args:
            url: The URL to crawl
            
        Returns:
            Dict containing mock crawled data
        """
        if not self.is_valid_url(url):
            return {
                "success": False,
                "error": "Invalid URL format",
                "url": url,
                "content": None
            }
        
        # Return mock content for testing
        mock_content = f"This is mock crawled content from {url}. In a real implementation, this would be the actual webpage content extracted using a web crawler."
        
        return {
            "success": True,
            "url": url,
            "content": mock_content,
            "title": f"Mock Title for {url}",
            "description": f"Mock description for the webpage at {url}",
            "word_count": len(mock_content.split()),
            "error": None
        }
    
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a database record for link crawling.
        
        Args:
            record: Database record containing message information
            
        Returns:
            Dict containing processing results with ID preserved
        """
        try:
            # Always preserve the ID field
            result = {"ID": record.get("ID")}
            
            # Check if content type is appropriate for link crawling
            content_type = record.get("content_type", "").lower()
            if content_type not in ["link", "text"]:
                logger.info(f"Skipping link crawling for content type: {content_type}")
                return result  # Return with just ID, no processing results
            
            # Extract URL from content (try both 'content' and 'raw_text' fields)
            content = record.get('content') or record.get('raw_text', '')
            
            if not content:
                result.update({
                    "link_crawl_results": {
                        "success": False,
                        "error": "No content found in record",
                        "crawl_status": "failed",
                        "crawl_reason": "No content found"
                    }
                })
                return result
            
            # Extract URLs from content
            urls = self.extract_urls_from_content(content)
            
            if not urls:
                # Check if content itself is a URL
                if self.is_valid_url(content.strip()):
                    urls = [content.strip()]
                else:
                    result.update({
                        "link_crawl_results": {
                            "success": False,
                            "error": "No valid URLs found in content",
                            "crawl_status": "failed",
                            "crawl_reason": "No valid URLs found"
                        }
                    })
                    return result
            
            # Crawl the first URL found
            url = urls[0]
            crawl_result = self.crawl_url(url)
            
            if crawl_result["success"]:
                # Update record with crawled content
                result.update({
                    "link_crawl_results": {
                        "success": True,
                        "crawled_content": crawl_result["content"],
                        "crawled_title": crawl_result.get("title", ""),
                        "crawled_description": crawl_result.get("description", ""),
                        "crawled_url": crawl_result["url"],
                        "crawled_word_count": crawl_result.get("word_count", 0),
                        "crawl_status": "success",
                        "crawl_reason": None
                    }
                })
                
                logger.info(f"Successfully processed link record {record.get('ID')}")
                return result
            else:
                result.update({
                    "link_crawl_results": {
                        "success": False,
                        "error": crawl_result["error"],
                        "crawl_status": "failed",
                        "crawl_reason": crawl_result["error"]
                    }
                })
                return result
                
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            return {
                "ID": record.get("ID"),
                "link_crawl_results": {
                    "success": False,
                    "error": str(e),
                    "crawl_status": "error",
                    "crawl_reason": f"Processing error: {str(e)}"
                }
            }

def main():
    """
    Main function to handle command line usage.
    Accepts JSON input from stdin or as command line argument.
    """
    try:
        # Initialize the link crawler
        crawler = SimpleLinkCrawler()
        
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
        
        # Process the record
        result = crawler.process_record(record)
        
        # Output results as JSON
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "link_crawl_results": {
                "success": False,
                "error": str(e),
                "crawl_status": "error"
            }
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
