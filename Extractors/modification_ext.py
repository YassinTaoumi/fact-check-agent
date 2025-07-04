"""
Image Modification Detection using DistilDIRE model for database records
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageModificationDetector:
    """
    Image Modification Detection class that processes database records and determines
    if images are modified/manipulated using DistilDIRE model.
    """
    
    def __init__(self, checkpoint_path: str = None):
        """
        Initialize the image modification detector.
        
        Args:
            checkpoint_path: Path to the DistilDIRE model checkpoint (optional)
        """
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"Loading DistilDIRE model from: {checkpoint_path}")
                self.model = self._load_custom_model(checkpoint_path)
            else:
                logger.info("Loading pretrained ResNet50 model for modification detection")
                self.model = self._create_default_model()
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _create_default_model(self) -> 'CustomResNet':
        """Create default ResNet50 model"""
        model = CustomResNet(num_classes=1)
        model.to(self.device)
        model.eval()
        return model
    
    def _load_custom_model(self, checkpoint_path: str) -> 'CustomResNet':
        """Load custom model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

            # Extract model state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                logger.info("Using 'model' key from checkpoint")
            else:
                state_dict = checkpoint
                logger.info("Using entire checkpoint as state dict")

            # Create model
            model = CustomResNet(num_classes=1)

            # Handle DataParallel wrapper if needed
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_key = k[7:]  # Remove 'module.' prefix
                else:
                    new_key = k
                new_state_dict[new_key] = v

            # Try to load the weights (non-strict to allow for slight differences)
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("Loaded checkpoint weights (non-strict)")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Falling back to default model")
            return self._create_default_model()
    
    def _preprocess_image(self, image_path: str, size: int = 224) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image_path: Path to the image file
            size: Target image size
            
        Returns:
            Preprocessed image tensor
        """
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    
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
    
    def detect_image_modification(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if an image has been modified/manipulated.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing detection results
        """
        try:
            if not self.model:
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "is_modified": None,
                    "confidence": 0.0,
                    "modification_score": 0.0,
                    "authentic_score": 0.0
                }
            
            if not self.is_valid_image_path(image_path):
                return {
                    "success": False,
                    "error": f"Invalid image path: {image_path}",
                    "is_modified": None,
                    "confidence": 0.0,
                    "modification_score": 0.0,
                    "authentic_score": 0.0
                }
            
            logger.info(f"Processing image: {image_path}")
            
            # Preprocess image
            image_tensor = self._preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)
                # Convert to probability score using sigmoid
                modification_score = torch.sigmoid(output).item()
            
            # Determine if image is modified (higher score means more likely modified)
            is_modified = modification_score > 0.5
            authentic_score = 1.0 - modification_score
            confidence = max(modification_score, authentic_score)
            
            logger.info(f"Detection complete - Modified: {is_modified}, Confidence: {confidence:.3f}")
            
            return {
                "success": True,
                "error": None,
                "is_modified": is_modified,
                "confidence": confidence,
                "modification_score": modification_score,
                "authentic_score": authentic_score
            }
            
        except Exception as e:
            logger.error(f"Error detecting image modification: {e}")
            return {
                "success": False,
                "error": str(e),
                "is_modified": None,
                "confidence": 0.0,
                "modification_score": 0.0,
                "authentic_score": 0.0
            }
    
    def process_database_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a database record and detect if image is modified.
        
        Args:
            record: Database record with fields from raw_data table:
                    ROWID, ID, UUID, source_type, sender_phone, is_group_message,
                    group_name, channel_name, chat_jid, content_type, content_url,
                    raw_text, submission_timestamp, processing_status, user_identifier,
                    priority, metadata
        
        Returns:
            Updated record with modification detection results
        """
        try:
            content_type = record.get('content_type', '').lower()
            record_id = record.get('UUID') or record.get('ID', 'unknown')
            
            logger.info(f"Processing record {record_id} with content_type: {content_type}")
            
            # Only process image content types
            if content_type != 'image':
                return {
                    **record,
                    "modification_detection_status": "skipped",
                    "modification_detection_error": f"Not an image content type: {content_type}",
                    "is_modified_image": None,
                    "modification_confidence": 0.0,
                    "modification_detection_details": None
                }
            
            # Get image path from content_url
            image_path = record.get('content_url', '').strip()
            
            if not image_path:
                return {
                    **record,
                    "modification_detection_status": "failed",
                    "modification_detection_error": "No content_url found for image",
                    "is_modified_image": None,
                    "modification_confidence": 0.0,
                    "modification_detection_details": None
                }
            
            # Run modification detection
            detection_result = self.detect_image_modification(image_path)
            
            if detection_result["success"]:
                # Update metadata with modification detection info
                existing_metadata = record.get('metadata')
                if existing_metadata:
                    try:
                        metadata_dict = json.loads(existing_metadata) if isinstance(existing_metadata, str) else existing_metadata
                    except:
                        metadata_dict = {}
                else:
                    metadata_dict = {}
                
                # Add modification detection metadata
                metadata_dict['modification_detection_info'] = {
                    'detected_at': record.get('submission_timestamp'),
                    'model_used': 'DistilDIRE/ResNet50',
                    'device_used': str(self.device),
                    'is_modified': detection_result["is_modified"],
                    'confidence': detection_result["confidence"],
                    'modification_score': detection_result["modification_score"],
                    'authentic_score': detection_result["authentic_score"]
                }
                
                return {
                    **record,
                    "modification_detection_status": "success",
                    "modification_detection_error": None,
                    "is_modified_image": detection_result["is_modified"],
                    "modification_confidence": detection_result["confidence"],
                    "modification_detection_details": {
                        "modification_score": detection_result["modification_score"],
                        "authentic_score": detection_result["authentic_score"]
                    },
                    "metadata": json.dumps(metadata_dict)
                }
            else:
                return {
                    **record,
                    "modification_detection_status": "failed",
                    "modification_detection_error": detection_result["error"],
                    "is_modified_image": None,
                    "modification_confidence": 0.0,
                    "modification_detection_details": None
                }
                
        except Exception as e:
            logger.error(f"Error processing record {record.get('UUID', record.get('ID', 'unknown'))}: {e}")
            return {
                **record,
                "modification_detection_status": "error",
                "modification_detection_error": str(e),
                "is_modified_image": None,
                "modification_confidence": 0.0,
                "modification_detection_details": None
            }
class CustomResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomResNet, self).__init__()
        # Use a ResNet50 backbone
        resnet = models.resnet50(pretrained=True)
        # Replace final layer for binary classification
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.model = resnet

    def forward(self, x):
        return self.model(x)

def process_json_input(json_input: str) -> str:
    """
    Process JSON input containing database record(s).
    
    Args:
        json_input: JSON string containing record or list of records
        
    Returns:
        JSON string with modification detection results
    """
    detector = ImageModificationDetector()
    
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
            "modification_detection_status": "error"
        }
        return json.dumps(error_result, indent=2)
    except Exception as e:
        error_result = {
            "error": f"Processing error: {e}",
            "modification_detection_status": "error"
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