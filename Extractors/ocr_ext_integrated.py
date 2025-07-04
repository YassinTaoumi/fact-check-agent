import json
import os
import sys
import logging
from typing import Dict, Any, Optional
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForImageTextToText,
)
import torch
from PIL import Image

# Add path to parent directory for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_normalizer import normalize_result, get_text_preview

# Try to import BitsAndBytesConfig, but handle gracefully if not available
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    logger = logging.getLogger(__name__)
    logger.warning("BitsAndBytesConfig not available - will use standard precision")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRExtractor:
    """
    OCR Text Extraction class that processes database records and extracts text
    from image files using advanced vision-language models.
    """
    
    def __init__(self, model_id: str = "nanonets/Nanonets-OCR-s"):
        """
        Initialize the OCR extractor.
        
        Args:
            model_id: Hugging Face model ID for OCR
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.prompt = """Extract the text from the above document as if you were reading it naturally.
Return tables as HTML and equations as LaTeX."""
        
        try:
            logger.info(f"Loading OCR model: {model_id}")
            self._load_model()
            logger.info("OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            self.model = None
    
    def _load_model(self):
        """Load the OCR model components with CPU/GPU fallback"""
        try:
            # Check if CUDA is available and working
            device_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {device_available}")
            logger.info(f"BitsAndBytesConfig available: {HAS_BITSANDBYTES}")
            
            if device_available and HAS_BITSANDBYTES:
                # Try to load with GPU optimization first
                try:
                    logger.info("Attempting to load model with GPU optimization...")
                    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
                    
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        quantization_config=bnb_cfg,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                    logger.info("Model loaded successfully with GPU optimization")
                    
                except Exception as gpu_error:
                    logger.warning(f"GPU loading failed: {gpu_error}")
                    logger.info("Falling back to CPU loading...")
                    device_available = False
            
            if not device_available or not HAS_BITSANDBYTES:
                # Load model for CPU or without quantization
                logger.info("Loading model for CPU...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    device_map="cpu",
                )
                logger.info("Model loaded successfully on CPU")
            
            self.model.eval()
            
            # Load tokenizer and processor (these work on both CPU and GPU)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
        except Exception as e:
            logger.error(f"Failed to load model even with CPU fallback: {e}")
            # Try a more basic loading approach as final fallback
            try:
                logger.info("Attempting basic model loading...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                logger.info("Model loaded with basic configuration")
            except Exception as final_error:
                logger.error(f"All model loading attempts failed: {final_error}")
                raise
    
    def is_valid_image_path(self, image_path: str) -> bool:
        """
        Check if the image path exists and is a valid image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if valid image path, False otherwise
        """
        if not image_path or not os.path.exists(image_path):
            return False
        
        # Check if it's a supported image format
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        ext = os.path.splitext(image_path)[1].lower()
        return ext in valid_extensions
    
    def extract_text_from_image(self, image_path: str, max_new_tokens: int = 4096) -> Dict[str, Any]:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dict containing OCR results
        """
        try:
            if not self.model:
                return {
                    "success": False,
                    "error": "OCR model not loaded",
                    "extracted_text": "",
                    "word_count": 0,
                    "confidence": 0.0,
                }
            
            if not self.is_valid_image_path(image_path):
                return {
                    "success": False,
                    "error": f"Invalid image path: {image_path}",
                    "extracted_text": "",
                    "word_count": 0,
                    "confidence": 0.0,
                }
            
            logger.info(f"Extracting text from image: {image_path}")
            
            # Load image
            img = Image.open(image_path)
            
            # Prepare messages for the vision-language model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": self.prompt},
                ]}
            ]
            
            # Apply chat template
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[prompt_text], images=[img], return_tensors="pt", padding=True
            ).to(self.model.device)
            
            # Generate text
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            
            # Decode generated text
            gen_ids = out[:, inputs.input_ids.shape[-1]:]
            extracted_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
              # Calculate basic metrics
            word_count = len(extracted_text.split()) if extracted_text else 0
            
            logger.info(f"OCR extraction completed. Word count: {word_count}")
            
            # Prepare result for normalization
            result = {
                "success": True,
                "extracted_text": extracted_text,
                "word_count": word_count,
                "confidence": 0.95,  # Model-based OCR typically has high confidence
                "model_used": self.model_id,
                "processing_method": "vision_language_model"
            }
            
            # Apply text normalization
            normalized_result = normalize_result(result, "extracted_text")
            
            return normalized_result
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "word_count": 0,
                "confidence": 0.0,
            }
    
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a database record for OCR extraction.
        
        Args:
            record: Database record containing message information
            
        Returns:
            Dict containing processing results with ID preserved
        """
        try:
            # Always preserve the ID field
            result = {"ID": record.get("ID")}
            
            # Check if content type is appropriate for OCR
            content_type = record.get("content_type", "").lower()
            if content_type not in ["image", "document"]:
                logger.info(f"Skipping OCR for content type: {content_type}")
                return result  # Return with just ID, no processing results
            
            # Get the content URL (file path)
            content_url = record.get("content_url")
            if not content_url:
                result.update({
                    "ocr_results": {
                        "success": False,
                        "error": "No content_url provided",
                        "extracted_text": "",
                        "word_count": 0,
                        "confidence": 0.0,
                    }
                })
                return result
            
            # Perform OCR extraction
            ocr_results = self.extract_text_from_image(content_url)
            result["ocr_results"] = ocr_results
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            return {
                "ID": record.get("ID"),
                "ocr_results": {
                    "success": False,
                    "error": str(e),
                    "extracted_text": "",
                    "word_count": 0,
                    "confidence": 0.0,
                }
            }

def main():
    """
    Main function to handle command line usage.
    Accepts JSON input from stdin or as command line argument.
    """
    try:
        # Initialize the OCR extractor
        extractor = OCRExtractor()
        
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
        result = extractor.process_record(record)
        
        # Output results as JSON
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "ocr_results": {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "word_count": 0,
                "confidence": 0.0,
            }
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
