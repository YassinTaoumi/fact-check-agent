import json
import os
import logging
from typing import Dict, Any, Optional
from transformers import pipeline
from PIL import Image
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArtificialImageDetector:
    """
    AI Image Detection class that processes database records and determines
    if images are artificial/AI-generated or real.
    """
    
    def __init__(self, model_name: str = "haywoodsloan/ai-image-detector-deploy"):
        """
        Initialize the AI image detector.
        
        Args:
            model_name: Hugging Face model name for AI image detection
        """
        try:
            logger.info(f"Loading AI image detection model: {model_name}")
            self.pipe = pipeline("image-classification", model=model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.pipe = None
    
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
    
    def detect_artificial_image(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if an image is artificial/AI-generated or real.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing detection results
        """
        try:
            if not self.pipe:
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "is_artificial": None,
                    "confidence": 0.0,
                    "raw_results": None
                }
            
            if not self.is_valid_image_path(image_path):
                return {
                    "success": False,
                    "error": f"Invalid image path: {image_path}",
                    "is_artificial": None,
                    "confidence": 0.0,
                    "raw_results": None
                }
            
            logger.info(f"Processing image: {image_path}")
            
            # Load and process the image
            image = Image.open(image_path)
            
            # Run AI detection inference
            results = self.pipe(image)
            
            # Parse results - typically returns list of dicts with 'label' and 'score'
            artificial_score = 0.0
            real_score = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'artificial' in label or 'ai' in label or 'generated' in label or 'fake' in label:
                    artificial_score = max(artificial_score, score)
                elif 'real' in label or 'human' in label or 'authentic' in label:
                    real_score = max(real_score, score)
            
            # Determine if image is artificial (higher score wins)
            is_artificial = artificial_score > real_score
            confidence = max(artificial_score, real_score)
            
            logger.info(f"Detection complete - Artificial: {is_artificial}, Confidence: {confidence:.3f}")
            
            return {
                "success": True,
                "error": None,
                "is_artificial": is_artificial,
                "confidence": confidence,
                "artificial_score": artificial_score,
                "real_score": real_score,
                "raw_results": results
            }
            
        except Exception as e:
            logger.error(f"Error detecting artificial image: {e}")
            return {
                "success": False,
                "error": str(e),
                "is_artificial": None,
                "confidence": 0.0,
                "raw_results": None
            }
    
    def process_database_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a database record and detect if image is artificial.
        
        Args:
            record: Database record with fields from raw_data table:
                    ROWID, ID, UUID, source_type, sender_phone, is_group_message,
                    group_name, channel_name, chat_jid, content_type, content_url,
                    raw_text, submission_timestamp, processing_status, user_identifier,
                    priority, metadata
        
        Returns:
            Updated record with AI detection results
        """
        try:
            content_type = record.get('content_type', '').lower()
            record_id = record.get('UUID') or record.get('ID', 'unknown')
            
            logger.info(f"Processing record {record_id} with content_type: {content_type}")
            
            # Only process image content types
            if content_type != 'image':
                return {
                    **record,
                    "ai_detection_status": "skipped",
                    "ai_detection_error": f"Not an image content type: {content_type}",
                    "is_artificial_image": None,
                    "ai_confidence": 0.0,
                    "ai_detection_details": None
                }
            
            # Get image path from content_url
            image_path = record.get('content_url', '').strip()
            
            if not image_path:
                return {
                    **record,
                    "ai_detection_status": "failed",
                    "ai_detection_error": "No content_url found for image",
                    "is_artificial_image": None,
                    "ai_confidence": 0.0,
                    "ai_detection_details": None
                }
            
            # Run AI detection
            detection_result = self.detect_artificial_image(image_path)
            
            if detection_result["success"]:
                # Update metadata with AI detection info
                existing_metadata = record.get('metadata')
                if existing_metadata:
                    try:
                        metadata_dict = json.loads(existing_metadata) if isinstance(existing_metadata, str) else existing_metadata
                    except:
                        metadata_dict = {}
                else:
                    metadata_dict = {}
                
                # Add AI detection metadata
                metadata_dict['ai_detection_info'] = {
                    'detected_at': record.get('submission_timestamp'),
                    'model_used': 'haywoodsloan/ai-image-detector-deploy',
                    'is_artificial': detection_result["is_artificial"],
                    'confidence': detection_result["confidence"],
                    'artificial_score': detection_result["artificial_score"],
                    'real_score': detection_result["real_score"]
                }
                
                return {
                    **record,
                    "ai_detection_status": "success",
                    "ai_detection_error": None,
                    "is_artificial_image": detection_result["is_artificial"],
                    "ai_confidence": detection_result["confidence"],
                    "ai_detection_details": detection_result["raw_results"],
                    "metadata": json.dumps(metadata_dict)
                }
            else:
                return {
                    **record,
                    "ai_detection_status": "failed",
                    "ai_detection_error": detection_result["error"],
                    "is_artificial_image": None,
                    "ai_confidence": 0.0,
                    "ai_detection_details": None
                }
                
        except Exception as e:
            logger.error(f"Error processing record {record.get('UUID', record.get('ID', 'unknown'))}: {e}")
            return {
                **record,
                "ai_detection_status": "error",
                "ai_detection_error": str(e),
                "is_artificial_image": None,
                "ai_confidence": 0.0,
                "ai_detection_details": None
            }

def process_json_input(json_input: str) -> str:
    """
    Process JSON input containing database record(s).
    
    Args:
        json_input: JSON string containing record or list of records
        
    Returns:
        JSON string with AI detection results
    """
    detector = ArtificialImageDetector()
    
    try:
        data = json.loads(json_input)
        
        if isinstance(data, list):
            # Process multiple records
            results = []
            for record in data:
                result = detector.process_database_record(record)
                results.append(result)
            return json.dumps(results, indent=2)
        else:
            # Process single record
            result = detector.process_database_record(data)
            return json.dumps(result, indent=2)
            
    except json.JSONDecodeError as e:
        error_result = {
            "error": f"Invalid JSON input: {e}",
            "ai_detection_status": "error"
        }
        return json.dumps(error_result, indent=2)
    except Exception as e:
        error_result = {
            "error": f"Processing error: {e}",
            "ai_detection_status": "error"
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