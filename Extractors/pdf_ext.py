import json
import os
import sys
import logging
from typing import Dict, Any, Optional, List
import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
import PyPDF2
from io import BytesIO

# Add path to parent directory for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.text_normalizer import normalize_result, get_text_preview

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    """
    PDF Text Extraction class that processes database records and extracts text
    from PDF files using multiple extraction methods for maximum accuracy.
    """
    
    def __init__(self):
        """Initialize the PDF text extractor."""
        logger.info("PDF Text Extractor initialized")
    
    def is_valid_pdf_path(self, pdf_path: str) -> bool:
        """
        Check if the PDF path exists and is a valid PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: True if valid PDF path, False otherwise
        """
        if not pdf_path or not os.path.exists(pdf_path):
            return False
        
        # Check if it's a PDF file
        if not pdf_path.lower().endswith('.pdf'):
            return False
            
        # Try to open as PDF to verify it's valid
        try:
            with open(pdf_path, 'rb') as file:
                # Read first few bytes to check PDF header
                header = file.read(4)
                return header == b'%PDF'
        except Exception:
            return False
    
    def extract_with_pymupdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text using PyMuPDF (fitz) - good for most PDFs.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text or None if failed
        """
        try:
            logger.info("Extracting text with PyMuPDF")
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
                text += "\n\n"  # Add page separator
            
            doc.close()
            return text.strip() if text.strip() else None
            
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            return None
    
    def extract_with_pdfplumber(self, pdf_path: str) -> Optional[str]:
        """
        Extract text using pdfplumber - excellent for tables and complex layouts.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text or None if failed
        """
        try:
            logger.info("Extracting text with pdfplumber")
            text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                        text += "\n\n"  # Add page separator
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return None
    
    def extract_with_pdfminer(self, pdf_path: str) -> Optional[str]:
        """
        Extract text using pdfminer - good for complex PDFs with detailed layout analysis.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text or None if failed
        """
        try:
            logger.info("Extracting text with pdfminer")
            # Configure layout analysis parameters for better text extraction
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                box_margin=0.5,
                detect_vertical=True
            )
            
            text = pdfminer_extract_text(pdf_path, laparams=laparams)
            return text.strip() if text and text.strip() else None
            
        except Exception as e:
            logger.warning(f"pdfminer extraction failed: {e}")
            return None
    
    def extract_with_pypdf2(self, pdf_path: str) -> Optional[str]:
        """
        Extract text using PyPDF2 - fallback method.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text or None if failed
        """
        try:
            logger.info("Extracting text with PyPDF2")
            text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                        text += "\n\n"  # Add page separator
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return None
    
    def extract_pdf_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using multiple methods for maximum accuracy.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict containing extraction results
        """
        try:
            if not self.is_valid_pdf_path(pdf_path):
                return {
                    "success": False,
                    "error": f"Invalid PDF path: {pdf_path}",
                    "extracted_text": None,
                    "text_length": 0,
                    "extraction_method": None,
                    "extraction_details": None
                }
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Try multiple extraction methods in order of preference
            extraction_methods = [
                ("pdfplumber", self.extract_with_pdfplumber),
                ("pymupdf", self.extract_with_pymupdf),
                ("pdfminer", self.extract_with_pdfminer),
                ("pypdf2", self.extract_with_pypdf2)
            ]
            
            best_text = None
            best_method = None
            extraction_attempts = {}
            
            for method_name, extraction_func in extraction_methods:
                try:
                    extracted_text = extraction_func(pdf_path)
                    
                    if extracted_text:
                        text_length = len(extracted_text)
                        extraction_attempts[method_name] = {
                            "success": True,
                            "text_length": text_length,
                            "preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
                        }
                        
                        # Use the first successful extraction with substantial content
                        if not best_text or text_length > len(best_text):
                            best_text = extracted_text
                            best_method = method_name
                    else:
                        extraction_attempts[method_name] = {
                            "success": False,
                            "text_length": 0,
                            "error": "No text extracted"
                        }
                        
                except Exception as e:                    extraction_attempts[method_name] = {
                        "success": False,
                        "text_length": 0,
                        "error": str(e)
                    }
            
            if best_text:
                logger.info(f"Successfully extracted {len(best_text)} characters using {best_method}")
                
                # Prepare result for normalization
                result = {
                    "success": True,
                    "error": None,
                    "extracted_text": best_text,
                    "text_length": len(best_text),
                    "extraction_method": best_method,
                    "extraction_details": extraction_attempts
                }
                
                # Apply text normalization
                normalized_result = normalize_result(result, "extracted_text")
                
                return normalized_result
            else:
                return {
                    "success": False,
                    "error": "All extraction methods failed",
                    "extracted_text": None,
                    "text_length": 0,
                    "extraction_method": None,
                    "extraction_details": extraction_attempts
                }
                
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": None,
                "text_length": 0,
                "extraction_method": None,
                "extraction_details": None
            }
    
    def process_database_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a database record and extract PDF text.
        
        Args:
            record: Database record with fields from raw_data table:
                    ROWID, ID, UUID, source_type, sender_phone, is_group_message,
                    group_name, channel_name, chat_jid, content_type, content_url,
                    raw_text, submission_timestamp, processing_status, user_identifier,
                    priority, metadata
          Returns:
            Updated record with PDF text extraction results
        """
        try:
            content_type = record.get('content_type', '').lower()
            record_id = record.get('UUID') or record.get('ID', 'unknown')
            
            logger.info(f"Processing record {record_id} with content_type: {content_type}")
            
            # Ensure ID is preserved in all outputs
            base_record = {
                "ROWID": record.get('ROWID'),
                "ID": record.get('ID'),
                "UUID": record.get('UUID'),
                **record  # Include all other fields
            }
              # Only process PDF content types
            if content_type != 'pdf':
                # Return nothing for non-PDF content types
                return None
              # Get PDF path from content_url
            pdf_path = record.get('content_url', '').strip()
            
            if not pdf_path:
                return {
                    **base_record,
                    "pdf_extraction_status": "failed",
                    "pdf_extraction_error": "No content_url found for PDF",
                    "pdf_extracted_text": None,
                    "pdf_text_length": 0,
                    "pdf_extraction_details": None
                }
            
            # Run PDF text extraction
            extraction_result = self.extract_pdf_text(pdf_path)
            
            if extraction_result["success"]:
                # Update metadata with PDF extraction info
                existing_metadata = record.get('metadata')
                if existing_metadata:
                    try:
                        metadata_dict = json.loads(existing_metadata) if isinstance(existing_metadata, str) else existing_metadata
                    except:
                        metadata_dict = {}
                else:
                    metadata_dict = {}
                
                # Add PDF extraction metadata
                metadata_dict['pdf_extraction_info'] = {
                    'extracted_at': record.get('submission_timestamp'),
                    'extraction_method': extraction_result["extraction_method"],
                    'text_length': extraction_result["text_length"],
                    'extraction_attempts': extraction_result["extraction_details"]
                }
                return {
                    **base_record,
                    "pdf_extraction_status": "success",
                    "pdf_extraction_error": None,
                    "pdf_extracted_text": extraction_result["extracted_text"],
                    "pdf_text_length": extraction_result["text_length"],
                    "pdf_extraction_details": {
                        "extraction_method": extraction_result["extraction_method"],
                        "extraction_attempts": extraction_result["extraction_details"]
                    },
                    "metadata": json.dumps(metadata_dict)
                }
            else:                return {
                    **base_record,
                    "pdf_extraction_status": "failed",
                    "pdf_extraction_error": extraction_result["error"],
                    "pdf_extracted_text": None,
                    "pdf_text_length": 0,
                    "pdf_extraction_details": extraction_result["extraction_details"]
                }
                
        except Exception as e:
            logger.error(f"Error processing record {record.get('UUID', record.get('ID', 'unknown'))}: {e}")
            # Ensure we have base_record structure even in error case
            base_record_error = {
                "ROWID": record.get('ROWID'),
                "ID": record.get('ID'),
                "UUID": record.get('UUID'),
                **record
            }
            return {
                **base_record_error,
                "pdf_extraction_status": "error",
                "pdf_extraction_error": str(e),
                "pdf_extracted_text": None,
                "pdf_text_length": 0,
                "pdf_extraction_details": None
            }

def process_json_input(json_input: str) -> str:
    """
    Process JSON input containing database record(s).
    
    Args:
        json_input: JSON string containing record or list of records
        
    Returns:
        JSON string with PDF extraction results
    """
    extractor = PDFTextExtractor()
    
    try:
        data = json.loads(json_input)
        
        if isinstance(data, list):
            # Process multiple records
            results = []
            for record in data:
                result = extractor.process_database_record(record)
                # Only add non-None results
                if result is not None:
                    results.append(result)
            
            # Return empty JSON array if no results
            if not results:
                return "[]"
            return json.dumps(results, indent=2)
        else:
            # Process single record
            result = extractor.process_database_record(data)
            # Return empty JSON object if None
            if result is None:
                return "{}"
            return json.dumps(result, indent=2)
            
    except json.JSONDecodeError as e:
        error_result = {
            "error": f"Invalid JSON input: {e}",
            "pdf_extraction_status": "error"
        }
        return json.dumps(error_result, indent=2)
    except Exception as e:
        error_result = {
            "error": f"Processing error: {e}",
            "pdf_extraction_status": "error"
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
