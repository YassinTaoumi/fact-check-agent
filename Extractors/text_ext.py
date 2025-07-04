import json
import os
import sys
import logging
import re
import unicodedata
import string
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add path to parent directory for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependencies for enhanced text processing
try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False 
    logger.warning("emoji library not installed - emoji handling will be limited")

try:
    import ftfy  # For fixing text encoding issues
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False
    logger.warning("ftfy library not installed - text encoding fixes will be limited")

from utils.text_normalizer import normalize_result, get_text_preview

class TextProcessor:
    """
    Text Processing class that cleans and processes text content from database records.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        logger.info("Text Processor initialized")
    
    def is_valid_text_record(self, record: Dict[str, Any]) -> bool:
        """
        Check if the record contains valid text content for processing.
        
        Args:
            record: Database record
            
        Returns:
            bool: True if valid text record, False otherwise
        """
        content_type = record.get('content_type', '').lower()
        
        # Check if content type is text
        if content_type != 'text':
            return False
        
        # Check if there's text content to process
        raw_text = record.get('raw_text', '')
        if not raw_text or not raw_text.strip():
            return False
        
        return True
    
    def clean_whitespace(self, text: str) -> str:
        """
        Clean and normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            str: Text with cleaned whitespace
        """
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading and trailing whitespace
        text = text.strip()
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up tabs and other whitespace characters
        text = text.replace('\t', ' ')
        text = text.replace('\r', '')
        
        return text
    
    def remove_unrecognized_characters(self, text: str) -> str:
        """
        Remove or replace unrecognized and problematic characters.
        
        Args:
            text: Input text
            
        Returns:
            str: Text with unrecognized characters cleaned        """
        # Fix encoding issues first if ftfy is available
        if HAS_FTFY:
            text = ftfy.fix_text(text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', '', text)
        
        # Remove or replace problematic Unicode characters
        text = re.sub(r'[\ufffd\ufffe\uffff]', '', text)  # Remove replacement/invalid characters
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def clean_special_characters(self, text: str, preserve_emojis: bool = True) -> str:
        """
        Clean special characters while optionally preserving emojis.
        
        Args:
            text: Input text
            preserve_emojis: Whether to preserve emoji characters
            
        Returns:
            str: Text with special characters cleaned        """
        if preserve_emojis and HAS_EMOJI:
            # Extract emojis to preserve them
            emojis = emoji.distinct_emoji_list(text)
            emoji_placeholder = "___EMOJI_{}___"
            emoji_map = {}
            
            # Replace emojis with placeholders
            for i, em in enumerate(emojis):
                placeholder = emoji_placeholder.format(i)
                emoji_map[placeholder] = em
                text = text.replace(em, placeholder)
        else:
            emoji_map = {}
        
        # Remove or clean special characters
        # Keep basic punctuation but remove excessive special chars
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\_\+\=\/\\\|\~\`\@\#\$\%\^\&\*]', ' ', text)
        
        # Clean up excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'[!]{2,}', '!', text)    # Multiple exclamation marks
        text = re.sub(r'[?]{2,}', '?', text)    # Multiple question marks
        text = re.sub(r'[-]{3,}', '---', text)  # Multiple dashes
        
        if preserve_emojis and HAS_EMOJI:
            # Restore emojis
            for placeholder, emoji_char in emoji_map.items():
                text = text.replace(placeholder, emoji_char)
        
        return text
    
    def extract_metadata(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """
        Extract metadata and statistics from the text.
        
        Args:
            original_text: Original text before cleaning
            cleaned_text: Text after cleaning
            
        Returns:
            Dict containing text metadata and statistics
        """
        metadata = {
            "original_length": len(original_text),
            "cleaned_length": len(cleaned_text),
            "reduction_ratio": 1 - (len(cleaned_text) / len(original_text)) if len(original_text) > 0 else 0,
            "word_count": len(cleaned_text.split()) if cleaned_text else 0,
            "sentence_count": len(re.split(r'[.!?]+', cleaned_text)) if cleaned_text else 0,
            "paragraph_count": len([p for p in cleaned_text.split('\n\n') if p.strip()]) if cleaned_text else 0,
            "character_stats": {
                "letters": sum(1 for c in cleaned_text if c.isalpha()),
                "digits": sum(1 for c in cleaned_text if c.isdigit()),
                "spaces": sum(1 for c in cleaned_text if c.isspace()),
                "punctuation": sum(1 for c in cleaned_text if c in string.punctuation)
            }
        }
        
        # Detect language hints
        if cleaned_text:
            # Simple language detection based on character patterns
            latin_chars = sum(1 for c in cleaned_text if ord(c) < 256)
            total_chars = len(cleaned_text)
            metadata["latin_ratio"] = latin_chars / total_chars if total_chars > 0 else 0
            
            # Detect if text contains URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, cleaned_text)
            metadata["contains_urls"] = len(urls) > 0
            metadata["url_count"] = len(urls)
              # Detect emojis (if emoji library is available)
            if HAS_EMOJI:
                emojis = emoji.distinct_emoji_list(cleaned_text)
                metadata["contains_emojis"] = len(emojis) > 0
                metadata["emoji_count"] = len(emojis)
            else:
                metadata["contains_emojis"] = False
                metadata["emoji_count"] = 0
            
            # Detect mentions and hashtags
            mentions = re.findall(r'@\w+', cleaned_text)
            hashtags = re.findall(r'#\w+', cleaned_text)
            metadata["mention_count"] = len(mentions)
            metadata["hashtag_count"] = len(hashtags)
        
        return metadata
    
    def process_text(self, text: str, preserve_emojis: bool = True, aggressive_cleaning: bool = False) -> Dict[str, Any]:
        """
        Process and clean text content.
        
        Args:
            text: Input text to process
            preserve_emojis: Whether to preserve emoji characters
            aggressive_cleaning: Whether to apply more aggressive cleaning
            
        Returns:
            Dict containing processed text and metadata
        """
        try:
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "Empty or invalid text input",
                    "original_text": text,
                    "cleaned_text": "",
                    "metadata": {}
                }
            
            original_text = text
            logger.info(f"Processing text of length: {len(original_text)}")
            
            # Step 1: Clean whitespace
            processed_text = self.clean_whitespace(text)
            
            # Step 2: Remove unrecognized characters
            processed_text = self.remove_unrecognized_characters(processed_text)
            
            # Step 3: Clean special characters
            processed_text = self.clean_special_characters(processed_text, preserve_emojis)
            
            # Step 4: Additional aggressive cleaning if requested
            if aggressive_cleaning:
                # Remove extra spaces between words
                processed_text = re.sub(r' {2,}', ' ', processed_text)
                # Remove standalone special characters
                processed_text = re.sub(r'\s+[^\w\s]\s+', ' ', processed_text)
            
            # Step 5: Final whitespace cleanup
            processed_text = self.clean_whitespace(processed_text)
              # Extract metadata
            metadata = self.extract_metadata(original_text, processed_text)
            
            logger.info(f"Text processing complete - reduced from {len(original_text)} to {len(processed_text)} characters")
            
            result = {
                "success": True,
                "error": None,
                "original_text": original_text,
                "extracted_text": processed_text,
                "metadata": metadata,
                "processing_info": {
                    "preserve_emojis": preserve_emojis,
                    "aggressive_cleaning": aggressive_cleaning,
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
            # Apply text normalization
            normalized_result = normalize_result(result, "extracted_text")
            
            # Convert back to expected format
            return {
                "success": True,
                "error": None,
                "original_text": original_text,
                "cleaned_text": normalized_result["extracted_text"],
                "metadata": normalized_result.get("metadata", metadata),
                "processing_info": {
                    "preserve_emojis": preserve_emojis,
                    "aggressive_cleaning": aggressive_cleaning,
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_text": text,
                "cleaned_text": "",
                "metadata": {}
            }
    
    def process_database_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a database record and return only cleaned text with ID.
        
        Args:
            record: Database record with fields from raw_data table
        
        Returns:
            Simple result with ID and cleaned text only
        """
        try:
            content_type = record.get('content_type', '').lower()
            record_id = record.get('ID') or record.get('UUID', 'unknown')
            
            logger.info(f"Processing record {record_id} with content_type: {content_type}")
            
            # Only process text content types
            if not self.is_valid_text_record(record):
                return {
                    "ID": record_id,
                    "cleaned_text": None,
                    "error": f"Not a text content type: {content_type}"
                }
            
            # Get text content
            raw_text = record.get('raw_text', '').strip()
            
            if not raw_text:
                return {
                    "ID": record_id,
                    "cleaned_text": None,
                    "error": "No text content to process"
                }
              # Clean the text using simple cleaning
            cleaned_text = raw_text.strip()
            
            # Normalize whitespace
            cleaned_text = ' '.join(cleaned_text.split())
            
            # Basic character cleaning - remove control characters except newlines/tabs
            cleaned_text = ''.join(char for char in cleaned_text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
            
            # Basic special character cleanup
            cleaned_text = re.sub(r'[.]{3,}', '...', cleaned_text)  # Multiple dots to ellipsis
            cleaned_text = re.sub(r'[!]{2,}', '!', cleaned_text)    # Multiple exclamation marks
            cleaned_text = re.sub(r'[?]{2,}', '?', cleaned_text)    # Multiple question marks
              # Normalize the result using the global text normalizer
            result = {
                "ID": record_id,
                "extracted_text": cleaned_text
            }
            
            # Apply text normalization (limit to 5 lines if content exceeds 4 lines)
            normalized_result = normalize_result(result, "extracted_text")
            
            # Convert back to expected format
            return {
                "ID": record_id,
                "cleaned_text": normalized_result["extracted_text"]
            }
                
        except Exception as e:
            logger.error(f"Error processing record {record.get('ID', record.get('UUID', 'unknown'))}: {e}")
            return {
                "ID": record.get('ID') or record.get('UUID', 'unknown'),
                "cleaned_text": None,
                "error": str(e)
            }

def process_json_input(json_input: str) -> str:
    """
    Process JSON input containing database record(s).
    
    Args:
        json_input: JSON string containing record or list of records
        
    Returns:
        JSON string with text processing results
    """
    processor = TextProcessor()
    
    try:
        data = json.loads(json_input)
        
        if isinstance(data, list):
            # Process multiple records
            results = []
            for record in data:
                result = processor.process_database_record(record)
                results.append(result)
            return json.dumps(results, indent=2)
        else:
            # Process single record
            result = processor.process_database_record(data)
            return json.dumps(result, indent=2)
            
    except json.JSONDecodeError as e:
        error_result = {
            "error": f"Invalid JSON input: {e}",
            "text_processing_status": "error"
        }
        return json.dumps(error_result, indent=2)
    except Exception as e:
        error_result = {
            "error": f"Processing error: {e}",
            "text_processing_status": "error"
        }
        return json.dumps(error_result, indent=2)

def main():
    """
    Main function for command-line usage and RabbitMQ workflow.
    """
    if len(sys.argv) > 1:
        # Read JSON from command-line argument
        json_input = sys.argv[1]
    else:
        # Read JSON from stdin for RabbitMQ workflow
        json_input = sys.stdin.read()
    
    # Process the JSON input
    result = process_json_input(json_input)
    
    # Output the result
    print(result)

if __name__ == "__main__":
    main()
