#!/usr/bin/env python3
"""
Global Text Normalizer Utility

This utility provides text normalization functions that can be used across all extractors
to standardize text output, including limiting content length and formatting.
"""

import re
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextNormalizer:
    """
    Global text normalizer for consistent text processing across extractors
    """
    
    def __init__(self, max_lines: int = 5, line_threshold: int = 4):
        """
        Initialize the text normalizer
        
        Args:
            max_lines: Maximum number of lines to keep when content exceeds threshold
            line_threshold: Number of lines that triggers truncation
        """
        self.max_lines = max_lines
        self.line_threshold = line_threshold
        logger.info(f"TextNormalizer initialized: max_lines={max_lines}, threshold={line_threshold}")
    
    def normalize_content(self, content: str, content_type: str = "text") -> str:
        """
        Normalize content by limiting lines when it exceeds the threshold
        
        Args:
            content: The text content to normalize
            content_type: Type of content (for logging purposes)
            
        Returns:
            str: Normalized content
        """
        if not content or not isinstance(content, str):
            return ""
        
        # Clean the content first
        cleaned_content = self._clean_text(content)
        
        # Split into lines
        lines = cleaned_content.split('\n')
        
        # Check if we need to truncate
        if len(lines) > self.line_threshold:
            truncated_lines = lines[:self.max_lines]
            result = '\n'.join(truncated_lines)
            
            logger.info(f"Content truncated from {len(lines)} to {len(truncated_lines)} lines for {content_type}")            
            # Add truncation indicator
            if len(lines) > self.max_lines:
                result += f"\n... [Content truncated: {len(lines) - self.max_lines} more lines]"
            
            return result
        
        return cleaned_content
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content while preserving line structure
        
        Args:
            text: Raw text content
            
        Returns:
            str: Cleaned text
        """
        # Normalize line breaks first (preserve line structure)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Remove excessive empty lines (but preserve some line breaks)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean whitespace within lines (but preserve line breaks)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive whitespace within each line
            cleaned_line = re.sub(r'\s+', ' ', line.strip())
            cleaned_lines.append(cleaned_line)
        
        # Rejoin lines and strip leading/trailing whitespace
        text = '\n'.join(cleaned_lines).strip()
        
        return text
    
    def normalize_extraction_result(self, result: Dict[str, Any], content_key: str = "extracted_text") -> Dict[str, Any]:
        """
        Normalize the extracted text in a result dictionary
        
        Args:
            result: Dictionary containing extraction results
            content_key: Key that contains the text content to normalize
            
        Returns:
            Dict[str, Any]: Result with normalized content
        """
        if content_key in result and result[content_key]:
            original_length = len(str(result[content_key]))
            result[content_key] = self.normalize_content(str(result[content_key]))
            new_length = len(result[content_key])
            
            # Add metadata about normalization
            if "metadata" not in result:
                result["metadata"] = {}
            
            result["metadata"].update({
                "text_normalized": True,
                "original_length": original_length,
                "normalized_length": new_length,
                "max_lines": self.max_lines,
                "line_threshold": self.line_threshold
            })
            
            if original_length != new_length:
                logger.info(f"Text normalized: {original_length} -> {new_length} characters")
        
        return result
    
    def get_preview(self, content: str, max_chars: int = 200) -> str:
        """
        Get a preview of content for logging/debugging
        
        Args:
            content: Text content
            max_chars: Maximum characters in preview
            
        Returns:
            str: Preview text
        """
        if not content:
            return "[Empty content]"
        
        preview = content.replace('\n', ' ').strip()
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "..."
        
        return preview


# Global instance for use across extractors
default_normalizer = TextNormalizer()


def normalize_text(content: str, content_type: str = "text") -> str:
    """
    Convenience function for quick text normalization
    
    Args:
        content: Text content to normalize
        content_type: Type of content
        
    Returns:
        str: Normalized text
    """
    return default_normalizer.normalize_content(content, content_type)


def normalize_result(result: Dict[str, Any], content_key: str = "extracted_text") -> Dict[str, Any]:
    """
    Convenience function for normalizing extraction results
    
    Args:
        result: Dictionary containing extraction results
        content_key: Key containing the text content
        
    Returns:
        Dict[str, Any]: Normalized result
    """
    return default_normalizer.normalize_extraction_result(result, content_key)


def get_text_preview(content: str, max_chars: int = 200) -> str:
    """
    Convenience function for getting text preview
    
    Args:
        content: Text content
        max_chars: Maximum characters in preview
        
    Returns:
        str: Preview text
    """
    return default_normalizer.get_preview(content, max_chars)


if __name__ == "__main__":
    # Test the normalizer
    test_content = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8"""
    
    print("Original content:")
    print(test_content)
    print(f"Lines: {len(test_content.split('\n'))}")
    
    normalized = normalize_text(test_content, "test")
    print("\nNormalized content:")
    print(normalized)
    print(f"Lines: {len(normalized.split('\n'))}")
    
    # Test with result dictionary
    result = {"extracted_text": test_content, "confidence": 0.95}
    normalized_result = normalize_result(result)
    print("\nNormalized result:")
    print(normalized_result)
