"""
RabbitMQ Worker Base Class and Specific Workers

This module provides:
1. Base worker class for RabbitMQ message processing
2. Specific workers for different content types (image, video, PDF, text)
3. Error handling and result publishing
"""

import os
import json
import sqlite3
import logging
import subprocess
import sys
import asyncio
import requests
import traceback
from typing import Dict, Any, Optional, List
import pika
from abc import ABC, abstractmethod
import traceback
from datetime import datetime

# Add fact-checker to path
fact_checker_path = os.path.join(os.path.dirname(__file__), 'fact-checker')
if fact_checker_path not in sys.path:
    sys.path.append(fact_checker_path)

# Import Qdrant integration
try:
    from qdrant_integration import store_fact_check_in_qdrant, qdrant_store
    QDRANT_AVAILABLE = True
    logging.info("✅ Qdrant integration loaded successfully")
except ImportError as e:
    QDRANT_AVAILABLE = False
    logging.warning(f"⚠️ Qdrant integration not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RabbitMQ Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

# Queue Names
RESULTS_QUEUE = "processing_results_queue"
FACT_CHECK_QUEUE = "fact_check_queue"

# Database Configuration
DB_PATH = os.getenv("DB_PATH", "databases/raw_data.db")

class BaseRabbitMQWorker(ABC):
    """
    Base class for RabbitMQ workers that process messages from queues
    """
    
    def __init__(self, queue_name: str, worker_name: str):
        """
        Initialize the worker
        
        Args:
            queue_name: Name of the queue to consume from
            worker_name: Name identifier for this worker
        """
        self.queue_name = queue_name
        self.worker_name = worker_name
        self.connection = None
        self.channel = None
        self.setup_rabbitmq()
    
    def setup_rabbitmq(self):
        """Establish RabbitMQ connection"""
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare the queue we'll consume from
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            self.channel.queue_declare(queue=RESULTS_QUEUE, durable=True)
            
            # Set quality of service (process one message at a time)
            self.channel.basic_qos(prefetch_count=1)
            
            logging.info(f"{self.worker_name} connected to RabbitMQ successfully")
            
        except Exception as e:
            logging.error(f"Failed to setup RabbitMQ for {self.worker_name}: {e}")
            raise
    
    def update_processing_status(self, message_id: str, status: str, error_message: Optional[str] = None):
        """
        Update processing status in database
        
        Args:
            message_id: ID of the message being processed
            status: New status ('processing', 'completed', 'failed')
            error_message: Optional error message for failed status
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                if error_message:
                    conn.execute("""
                        UPDATE raw_data 
                        SET processing_status = ?, metadata = json_set(
                            COALESCE(metadata, '{}'), 
                            '$.error_message', ?
                        )
                        WHERE ID = ?
                    """, (status, error_message, message_id))
                else:
                    conn.execute("""
                        UPDATE raw_data 
                        SET processing_status = ?
                        WHERE ID = ?
                    """, (status, message_id))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Failed to update status for {message_id}: {e}")
    
    def publish_result(self, message_id: str, result: Dict[str, Any]):
        """
        Publish processing result to results queue
        
        Args:
            message_id: ID of the processed message
            result: Processing result dictionary
        """
        try:
            result_data = {
                "message_id": message_id,
                "worker_name": self.worker_name,
                "timestamp": datetime.now().isoformat(),
                "result": result
            }
            
            self.channel.basic_publish(
                exchange='',
                routing_key=RESULTS_QUEUE,
                body=json.dumps(result_data),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            logging.info(f"Result published for message {message_id}")
            
        except Exception as e:
            logging.error(f"Failed to publish result for {message_id}: {e}")
    
    @abstractmethod
    def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to process message data
        
        Args:
            message_data: Dictionary containing message data from database
            
        Returns:
            Dict[str, Any]: Processing result
        """
        pass
    
    def callback(self, ch, method, properties, body):
        """
        RabbitMQ callback function to process messages
        
        Args:
            ch: Channel object
            method: Method object
            properties: Properties object
            body: Message body
        """
        message_id = None
        try:
            # Parse message data
            message_data = json.loads(body)
            message_id = message_data.get('ID')
            
            logging.info(f"{self.worker_name} processing message: {message_id}")
            
            # Update status to processing
            self.update_processing_status(message_id, 'processing')
            
            # Process the message
            result = self.process_message(message_data)
            
            # Update status to completed
            self.update_processing_status(message_id, 'completed')
            
            # Publish result
            self.publish_result(message_id, result)
            
            # Store results in the processed table
            self.store_results_in_db(message_id, result)

            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            logging.info(f"{self.worker_name} completed processing message: {message_id}")
            
        except Exception as e:
            error_message = f"Processing failed: {str(e)}"
            logging.error(f"{self.worker_name} failed to process message {message_id}: {error_message}")
            logging.error(traceback.format_exc())
            
            # Update status to failed
            if message_id:
                self.update_processing_status(message_id, 'failed', error_message)
            
            # Acknowledge message to remove it from queue (or reject based on policy)
            ch.basic_ack(delivery_tag=method.delivery_tag)
    
    def start_consuming(self):
        """Start consuming messages from the queue"""
        logging.info(f"{self.worker_name} starting to consume from queue: {self.queue_name}")
        
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.callback
        )
        
        try:            self.channel.start_consuming()
        except KeyboardInterrupt:
            logging.info(f"{self.worker_name} stopping...")
            self.channel.stop_consuming()
            self.connection.close()

    def store_results_in_db(self, message_id: str, result: Dict[str, Any]):
        """
        Store processing results in the processed table

        Args:
            message_id: ID of the message being processed
            result: Dictionary containing processing results
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Map worker results to processed table columns
            processed_data = {
                "processing_status": "completed"
            }
            
            # Map text processing results
            if "text_cleaning" in result:
                text_result = result["text_cleaning"]
                if isinstance(text_result, dict) and "cleaned_text" in text_result:
                    processed_data["cleaned_text"] = text_result["cleaned_text"]
            
            # Map image processing results
            if "ocr" in result:
                ocr_result = result["ocr"]
                if isinstance(ocr_result, dict):
                    processed_data["ocr_text"] = json.dumps(ocr_result)
            
            if "ai_detection" in result:
                processed_data["ai_image_detection"] = json.dumps(result["ai_detection"])
            
            if "modification_detection" in result:
                processed_data["image_modification_detection"] = json.dumps(result["modification_detection"])
            
            # Map video processing results
            if "video_transcription" in result:
                processed_data["video_transcription"] = json.dumps(result["video_transcription"])
            
            # Map PDF processing results  
            if "pdf_text" in result:
                processed_data["pdf_text_extraction"] = json.dumps(result["pdf_text"])
            
            # Map link processing results
            if "link_crawling" in result:
                processed_data["link_content"] = json.dumps(result["link_crawling"])
            
            # Add processing metadata
            if "processing_timestamp" in result:
                processed_data["processing_timestamp"] = result["processing_timestamp"]
            
            if "error" in result:
                processed_data["error_message"] = str(result["error"])
                processed_data["processing_status"] = "failed"

            # Build SQL query
            if processed_data:
                columns = ', '.join(processed_data.keys())
                placeholders = ', '.join(['?'] * len(processed_data))
                sql = f"INSERT INTO processed (message_id, {columns}) VALUES (?, {placeholders})"

                # Execute query
                cursor.execute(sql, [message_id] + list(processed_data.values()))
                conn.commit()
                logging.info(f"Results stored in database for message: {message_id}")
            else:
                logging.warning(f"No valid results to store for message: {message_id}")
            
            conn.close()

        except Exception as e:
            logging.error(f"Failed to store results for {message_id}: {e}")
            logging.error(f"Result data: {result}")

class ImageProcessingWorker(BaseRabbitMQWorker):
    """Worker for processing image content (OCR, modification detection, AI detection)"""
    
    def __init__(self):
        super().__init__("image_processing_queue", "ImageProcessor")
    
    def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image messages through OCR, modification detection, and AI detection
        
        Args:
            message_data: Message data from database
            
        Returns:
            Dict[str, Any]: Combined processing results
        """
        results = {
            "content_type": "image",
            "processing_timestamp": datetime.now().isoformat(),
            "original_message_id": message_data.get('ID')
        }
        
        # Check if content type is actually image
        if message_data.get('content_type', '').lower() != 'image':
            results['error'] = f"Expected image content, got {message_data.get('content_type')}"
            return results
        
        try:
            # Run modification detection
            results['modification_detection'] = self.run_modification_detection(message_data)
            
            # Run AI image detection
            results['ai_detection'] = self.run_ai_detection(message_data)
              # Run OCR processing
            results['ocr'] = self.run_ocr_extraction(message_data)
            
            results['status'] = 'success'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logging.error(f"Image processing failed: {e}")
        
        return results
    
    def run_modification_detection(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run modification detection extractor"""
        try:
            extractor_path = os.path.join("Extractors", "modification_ext.py")
            if not os.path.exists(extractor_path):
                return {"error": "modification_ext.py not found"}
            
            # Run the extractor
            result = subprocess.run(
                [sys.executable, extractor_path],
                input=json.dumps(message_data),
                text=True,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"Modification detection failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"Failed to run modification detection: {str(e)}"}
    
    def run_ai_detection(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI image detection extractor"""
        try:
            extractor_path = os.path.join("Extractors", "artificial_image_ext.py")
            if not os.path.exists(extractor_path):
                return {"error": "artificial_image_ext.py not found"}
            
            # Run the extractor
            result = subprocess.run(
                [sys.executable, extractor_path],
                input=json.dumps(message_data),
                text=True,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"AI detection failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"Failed to run AI detection: {str(e)}"}
    
    def run_ocr_extraction(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR text extraction"""
        try:
            extractor_path = os.path.join("Extractors", "ocr_ext_integrated.py")
            if not os.path.exists(extractor_path):
                return {"error": "ocr_ext_integrated.py not found"}
            
            # Run the extractor
            result = subprocess.run(
                [sys.executable, extractor_path],
                input=json.dumps(message_data),
                text=True,
                capture_output=True,
                timeout=120  # OCR might take longer
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"OCR extraction failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"Failed to run OCR extraction: {str(e)}"}

class VideoProcessingWorker(BaseRabbitMQWorker):
    """Worker for processing video and audio content (transcription)"""
    
    def __init__(self):
        super().__init__("video_processing_queue", "VideoProcessor")
    
    def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video/audio messages through transcription
        
        Args:
            message_data: Message data from database
            
        Returns:
            Dict[str, Any]: Transcription results
        """
        results = {
            "content_type": message_data.get('content_type'),
            "processing_timestamp": datetime.now().isoformat(),
            "original_message_id": message_data.get('ID')
        }
        
        # Check if content type is video or audio
        content_type = message_data.get('content_type', '').lower()
        if content_type not in ['video', 'audio']:
            results['error'] = f"Expected video/audio content, got {content_type}"
            return results
        
        try:
            # Run video transcription
            results.update(self.run_video_transcription(message_data))
            results['status'] = 'success'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logging.error(f"Video processing failed: {e}")
        
        return results
    
    def run_video_transcription(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run video transcription extractor"""
        try:
            extractor_path = os.path.join("Extractors", "video_transcriber.py")
            if not os.path.exists(extractor_path):
                return {"error": "video_transcriber.py not found"}
            
            # Run the extractor
            result = subprocess.run(
                [sys.executable, extractor_path],
                input=json.dumps(message_data),
                text=True,
                capture_output=True,
                timeout=300  # 5 minutes for video processing
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"Video transcription failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"Failed to run video transcription: {str(e)}"}

class PDFProcessingWorker(BaseRabbitMQWorker):
    """Worker for processing PDF content (text extraction)"""
    
    def __init__(self):
        super().__init__("pdf_processing_queue", "PDFProcessor")
    
    def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process PDF messages through text extraction
        
        Args:
            message_data: Message data from database
            
        Returns:
            Dict[str, Any]: PDF extraction results
        """
        results = {
            "content_type": "pdf", 
            "processing_timestamp": datetime.now().isoformat(),
            "original_message_id": message_data.get('ID')
        }
        
        # Check if content type is PDF or document
        content_type = message_data.get('content_type', '').lower()
        if content_type not in ['pdf', 'document']:
            results['error'] = f"Expected PDF/document content, got {content_type}"
            return results
        
        try:
            # Run PDF text extraction
            results.update(self.run_pdf_extraction(message_data))
            results['status'] = 'success'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logging.error(f"PDF processing failed: {e}")
        
        return results
    
    def run_pdf_extraction(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run PDF text extraction"""
        try:
            extractor_path = os.path.join("Extractors", "pdf_ext.py")
            if not os.path.exists(extractor_path):
                return {"error": "pdf_ext.py not found"}
            
            # Run the extractor
            result = subprocess.run(
                [sys.executable, extractor_path],
                input=json.dumps(message_data),
                text=True,
                capture_output=True,
                timeout=120  # 2 minutes for PDF processing
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"PDF extraction failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"Failed to run PDF extraction: {str(e)}"}

class TextProcessingWorker(BaseRabbitMQWorker):
    """Worker for processing text content (link crawling, fact checking)"""
    
    def __init__(self):
        super().__init__("text_processing_queue", "TextProcessor")
    
    def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text messages through link crawling and fact checking
        
        Args:
            message_data: Message data from database
            
        Returns:
            Dict[str, Any]: Text processing results
        """
        results = {
            "content_type": message_data.get('content_type'),
            "processing_timestamp": datetime.now().isoformat(),
            "original_message_id": message_data.get('ID')
        }
        
        # Check if content type is text or link
        content_type = message_data.get('content_type', '').lower()
        if content_type not in ['text', 'link']:
            results['error'] = f"Expected text/link content, got {content_type}"
            return results        
        try:
            # Run text cleaning and analysis
            results['text_cleaning'] = self.run_text_cleaning(message_data)
            
            # Run link crawling if applicable
            if content_type == 'link' or self.contains_links(message_data.get('raw_text', '')):
                results['link_crawling'] = self.run_link_crawling(message_data)
            
            # Run fact checking
            results['fact_checking'] = self.run_fact_checking(message_data)
            
            results['status'] = 'success'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logging.error(f"Text processing failed: {e}")
        
        return results
    
    def contains_links(self, text: str) -> bool:
        """Check if text contains URLs"""
        import re
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return bool(re.search(url_pattern, text))
    
    def run_link_crawling(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run link crawling extractor"""
        try:
            extractor_path = os.path.join("Extractors", "link_crawler.py")
            if not os.path.exists(extractor_path):
                return {"error": "link_crawler.py not found"}
            
            # Run the extractor
            result = subprocess.run(
                [sys.executable, extractor_path],
                input=json.dumps(message_data),
                text=True,
                capture_output=True,
                timeout=180  # 3 minutes for link crawling
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"Link crawling failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"Failed to run link crawling: {str(e)}"}
    
    def run_text_cleaning(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run text cleaning extractor"""
        try:
            extractor_path = os.path.join("Extractors", "text_ext.py")
            if not os.path.exists(extractor_path):
                return {"error": "text_ext.py not found"}
            
            # Run the extractor
            result = subprocess.run(
                [sys.executable, extractor_path],
                input=json.dumps(message_data),
                text=True,
                capture_output=True,
                timeout=60  # 1 minute for text cleaning
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"Text cleaning failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"Failed to run text cleaning: {str(e)}"}
    
    def run_fact_checking(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for fact checking - to be implemented later"""
        return {
            "status": "not_implemented", 
            "message": "Fact checking to be integrated later",
            "ID": message_data.get("ID")
        }


class FactCheckingWorker(BaseRabbitMQWorker):
    """Worker for fact-checking processed text content"""
    
    def __init__(self):
        super().__init__("fact_check_queue", "FactChecker")
        self.fact_checker = None
        self._setup_fact_checker()
    
    def _setup_fact_checker(self):
        """Initialize the fact-checker"""
        try:
            # Import fact checker after adding to path
            from fact_checker import fact_check_text
            self.fact_check_function = fact_check_text
            logging.info("Fact-checker initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize fact-checker: {e}")
            self.fact_check_function = None
    
    def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process messages that have completed initial processing and need fact-checking
        
        Args:
            message_data: Message data from database
            
        Returns:
            Dict[str, Any]: Fact-checking results
        """
        results = {
            "content_type": "fact_check",
            "processing_timestamp": datetime.now().isoformat(),
            "original_message_id": message_data.get('ID')
        }
        
        if not self.fact_check_function:
            results['status'] = 'error'
            results['error'] = 'Fact-checker not initialized'
            return results
        
        try:
            # Get the text to fact-check from processed results
            text_to_check = self._extract_text_for_fact_checking(message_data)
            
            if not text_to_check:
                results['status'] = 'skipped'
                results['reason'] = 'No text content available for fact-checking'
                return results
            
            # Run fact-checking asynchronously
            start_time = datetime.now()
            
            # Detect language from the text to be fact-checked
            try:
                from language_utils import detect_language
                detected_language = detect_language(text_to_check['text'])
                logging.info(f"Detected language for fact-checking: {detected_language}")
            except Exception as e:
                logging.error(f"Language detection failed: {e}")
                detected_language = "en"  # Default to English
            
            # Run fact-checking with language parameter
            fact_check_results = asyncio.run(self.fact_check_function(text_to_check['text'], language=detected_language))
            
            # Store fact-checking results in database
            self._store_fact_check_results(
                message_data.get('ID'),
                text_to_check,
                fact_check_results,
                start_time
            )
            
            # Send WhatsApp notification to original sender
            self._send_fact_check_notification(
                message_data.get('ID'),
                fact_check_results
            )
            
            results['status'] = 'success'
            results['fact_check_completed'] = True
            results['processing_duration'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logging.error(f"Fact-checking failed for message {message_data.get('ID')}: {e}")
            logging.error(traceback.format_exc())
        
        return results
    
    def _extract_text_for_fact_checking(self, message_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Extract text content from processed results for fact-checking
        
        Args:
            message_data: Original message data from raw_data table
            
        Returns:
            Dict with 'text' and 'source' keys, or None if no text found
        """
        try:
            message_id = message_data.get('ID')
            
            # Get processed results from database
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        cleaned_text, ocr_text, video_transcription, 
                        pdf_text_extraction, link_content
                    FROM processed 
                    WHERE message_id = ?
                """, (message_id,))
                
                row = cursor.fetchone()
                if not row:
                    # Check if there's raw text in the original message
                    cursor.execute("SELECT raw_text, content_type FROM raw_data WHERE ID = ?", (message_id,))
                    raw_row = cursor.fetchone()
                    if raw_row and raw_row[0] and raw_row[1] == 'text':
                        return {'text': raw_row[0], 'source': 'raw_text'}
                    return None
                
                cleaned_text, ocr_text, video_transcription, pdf_text_extraction, link_content = row
                
                # Priority order: cleaned_text -> pdf_text -> video_transcription -> ocr_text -> link_content
                if cleaned_text:
                    return {'text': cleaned_text, 'source': 'cleaned_text'}
                
                if pdf_text_extraction:
                    # PDF text might be JSON, extract it
                    try:
                        pdf_data = json.loads(pdf_text_extraction)
                        if isinstance(pdf_data, dict) and 'text' in pdf_data:
                            return {'text': pdf_data['text'], 'source': 'pdf_extraction'}
                        elif isinstance(pdf_data, str):
                            return {'text': pdf_data, 'source': 'pdf_extraction'}
                    except:
                        if pdf_text_extraction:
                            return {'text': pdf_text_extraction, 'source': 'pdf_extraction'}
                
                if video_transcription:
                    # Video transcription might be JSON
                    try:
                        video_data = json.loads(video_transcription)
                        if isinstance(video_data, dict) and 'transcription' in video_data:
                            return {'text': video_data['transcription'], 'source': 'transcription'}
                        elif isinstance(video_data, str):
                            return {'text': video_data, 'source': 'transcription'}
                    except:
                        if video_transcription:
                            return {'text': video_transcription, 'source': 'transcription'}
                
                if ocr_text:
                    # OCR text might be JSON
                    try:
                        ocr_data = json.loads(ocr_text)
                        if isinstance(ocr_data, dict) and 'text' in ocr_data:
                            return {'text': ocr_data['text'], 'source': 'ocr'}
                        elif isinstance(ocr_data, str):
                            return {'text': ocr_data, 'source': 'ocr'}
                    except:
                        if ocr_text:
                            return {'text': ocr_text, 'source': 'ocr'}
                
                if link_content:
                    # Link content might be JSON
                    try:
                        link_data = json.loads(link_content)
                        if isinstance(link_data, dict) and 'content' in link_data:
                            return {'text': link_data['content'], 'source': 'link_content'}
                        elif isinstance(link_data, str):
                            return {'text': link_data, 'source': 'link_content'}
                    except:
                        if link_content:
                            return {'text': link_content, 'source': 'link_content'}
                
                return None
                
        except Exception as e:
            logging.error(f"Failed to extract text for fact-checking: {e}")
            return None
    
    def _store_fact_check_results(self, message_id: str, text_info: Dict[str, str], 
                                fact_check_results: Dict[str, Any], start_time: datetime):
        """
        Store fact-checking results in the fact_check_results table
        
        Args:
            message_id: Original message ID
            text_info: Dict with text and source information
            fact_check_results: Results from fact-checker
            start_time: When fact-checking started
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                
                # Extract data from fact-check results
                input_text = text_info['text']
                input_source = text_info['source']
                
                # Handle both successful and error results
                if fact_check_results.get('success', False):
                    # Extract successful results
                    results = fact_check_results.get('results', [])
                    overall_stats = fact_check_results.get('overall_statistics', {})
                    metadata = fact_check_results.get('metadata', {})
                    
                    # Extract claims
                    claims = [result.get('claim', '') for result in results]
                    claims_json = json.dumps(claims)
                    num_claims = len(claims)
                    
                    # Extract search queries and URLs from metadata
                    search_queries = []
                    crawled_urls = []
                    for result in results:
                        if 'search_queries' in result:
                            search_queries.extend(result['search_queries'])
                        if 'sources' in result:
                            for source in result['sources']:
                                if 'url' in source:
                                    crawled_urls.append(source['url'])
                    
                    search_queries_json = json.dumps(list(set(search_queries)))
                    crawled_urls_json = json.dumps(list(set(crawled_urls)))
                    num_sources = len(set(crawled_urls))
                    
                    # Extract summaries
                    summaries = {}
                    for i, result in enumerate(results):
                        if 'sources' in result:
                            summaries[f"claim_{i}"] = result['sources']
                    summaries_json = json.dumps(summaries)
                    
                    # Extract individual claim verdicts
                    claim_verdicts = []
                    for result in results:
                        claim_verdicts.append({
                            'claim': result.get('claim', ''),
                            'verdict': result.get('verdict', ''),
                            'confidence': result.get('confidence', 0.0),
                            'justification': result.get('justification', '')
                        })
                    claim_verdicts_json = json.dumps(claim_verdicts)
                    
                    # Extract overall verdict
                    overall_verdict = overall_stats.get('overall_verdict', 'INSUFFICIENT_INFO')
                    overall_confidence = overall_stats.get('overall_confidence', 0.0)
                    overall_reasoning = overall_stats.get('overall_reasoning', '')
                    
                    num_llm_requests = metadata.get('total_llm_requests', 0)
                    error_message = None
                    
                else:
                    # Handle error case
                    claims_json = json.dumps([])
                    num_claims = 0
                    search_queries_json = json.dumps([])
                    crawled_urls_json = json.dumps([])
                    num_sources = 0
                    summaries_json = json.dumps({})
                    claim_verdicts_json = json.dumps([])
                    overall_verdict = 'INSUFFICIENT_INFO'
                    overall_confidence = 0.0
                    overall_reasoning = 'Fact-checking failed'
                    num_llm_requests = 0
                    error_message = fact_check_results.get('error', 'Unknown error')
                
                processing_duration = (datetime.now() - start_time).total_seconds()
                
                # Insert into fact_check_results table
                cursor.execute("""
                    INSERT INTO fact_check_results (
                        message_id, input_text, input_source,
                        claims_json, num_claims,
                        search_queries_json, crawled_urls_json, num_sources,
                        summaries_json, claim_verdicts_json,
                        overall_verdict, overall_confidence, overall_reasoning,
                        processing_duration_seconds, num_llm_requests, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message_id, input_text, input_source,
                    claims_json, num_claims,
                    search_queries_json, crawled_urls_json, num_sources,
                    summaries_json, claim_verdicts_json,
                    overall_verdict, overall_confidence, overall_reasoning,
                    processing_duration, num_llm_requests, error_message
                ))
                
                conn.commit()
                logging.info(f"Fact-check results stored for message {message_id}")
                
                # Store in Qdrant vector database
                if QDRANT_AVAILABLE and fact_check_results.get('success', False):
                    try:
                        # Store individual claim verdicts in Qdrant
                        stored_count = store_fact_check_in_qdrant(claim_verdicts, message_id)
                        logging.info(f"✅ Stored {stored_count} claims in Qdrant vector database")
                    except Exception as e:
                        logging.error(f"❌ Failed to store in Qdrant: {e}")
                        # Don't fail the whole process if Qdrant storage fails
                
        except Exception as e:
            logging.error(f"Failed to store fact-check results: {e}")
            logging.error(traceback.format_exc())
    
    def _send_fact_check_notification(self, message_id: str, fact_check_results: Dict[str, Any]):
        """
        Send WhatsApp notification to original sender with fact-check results
        
        Args:
            message_id: Original message ID
            fact_check_results: Results from fact-checker
        """
        try:
            # Get original sender information from raw_data table
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT sender_phone, raw_text, chat_jid, submission_timestamp
                    FROM raw_data 
                    WHERE ID = ?
                """, (message_id,))
                
                row = cursor.fetchone()
                if not row:
                    logging.error(f"Original message {message_id} not found for notification")
                    return
                
                sender_phone, original_text, chat_jid, submission_timestamp = row
            
            # Format notification message
            notification_message = self._format_notification_message(
                fact_check_results, 
                original_text, 
                submission_timestamp
            )
            
            # Send WhatsApp message
            success = self._send_whatsapp_message(sender_phone, chat_jid, notification_message)
            
            if success:
                logging.info(f"Fact-check notification sent to {sender_phone} for message {message_id}")
            else:
                logging.error(f"Failed to send fact-check notification to {sender_phone}")
                
        except Exception as e:
            logging.error(f"Failed to send fact-check notification: {e}")
            logging.error(traceback.format_exc())
    
    def _format_notification_message(self, fact_check_results: Dict[str, Any], 
                                   original_text: str, submission_timestamp: str) -> str:
        """
        Format the fact-check results into a user-friendly WhatsApp message
        
        Args:
            fact_check_results: Results from fact-checker
            original_text: Original message text
            submission_timestamp: When original message was submitted
            
        Returns:
            Formatted notification message
        """
        try:
            # Determine language for notification from fact_check_results metadata
            from language_utils import get_language_templates
            metadata = fact_check_results.get('metadata', {}) or {}
            detected_language = metadata.get('language')
            if not detected_language:
                # Fallback to auto-detect on original text
                from language_utils import detect_language
                detected_language = detect_language(original_text)
            logging.info(f"Using language for notification: {detected_language}")
            lang_templates = get_language_templates(detected_language)
            
            # Handle successful fact-check
            if fact_check_results.get('success', False):
                logging.info(f"Processing successful fact-check results for notification")
                overall_stats = fact_check_results.get('overall_statistics', {})
                overall_verdict = overall_stats.get('overall_verdict', 'UNVERIFIED')
                overall_confidence = overall_stats.get('overall_confidence', 0.0)
                overall_reasoning = overall_stats.get('overall_reasoning', 'No reasoning provided')
                
                logging.info(f"Overall verdict: {overall_verdict}, confidence: {overall_confidence}")
                
                # Get individual claim results
                claim_results = fact_check_results.get('results', [])
                logging.info(f"Number of claim results: {len(claim_results)}")
                
                # Create verdict emoji and description
                verdict_info = self._get_verdict_info(overall_verdict, overall_confidence, detected_language)
                
                # Format detailed analysis with sources and claims
                detailed_analysis = self._format_detailed_analysis(claim_results, overall_stats, detected_language)
                logging.info(f"Detailed analysis formatted, length: {len(detailed_analysis)} chars")
                
                # Check if we need to add a language note
                language_note = ""
                if detected_language != "en":
                    # Check if any of the claim results contain English content
                    has_english_content = any(
                        self._is_english_text(result.get('justification', '')) 
                        for result in claim_results
                    )
                    
                    if has_english_content:
                        if detected_language == "ar":
                            language_note = "\n_ملاحظة: بعض المحتوى مترجم تلقائياً. نعمل على تحسين الاستجابة بلغتك._\n"
                        elif detected_language == "fr":
                            language_note = "\n_Note : Une partie du contenu est traduite automatiquement. Nous travaillons à améliorer les réponses dans votre langue._\n"
                        elif detected_language == "es":
                            language_note = "\n_Nota: Parte del contenido está traducido automáticamente. Estamos trabajando para mejorar las respuestas en su idioma._\n"
                
                # Language-specific text
                if detected_language == "ar":
                    fact_check_title = "🔍 **نتائج التحقق من الحقائق**"
                    your_message_label = "📝 **رسالتك:**"
                    overall_verdict_label = "📊 **النتيجة الإجمالية:**"
                    confidence_label = "📊 **مستوى الثقة:**"
                    checked_at_label = "⏰ تم التحقق في:"
                    footer_text = "_تم إجراء هذا التحقق من الحقائق تلقائياً باستخدام مصادر متعددة وتحليل الذكاء الاصطناعي._"
                elif detected_language == "fr":
                    fact_check_title = "🔍 **Résultats de vérification des faits**"
                    your_message_label = "📝 **Votre message :**"
                    overall_verdict_label = "📊 **Verdict global :**"
                    confidence_label = "📊 **Confiance :**"
                    checked_at_label = "⏰ Vérifié à :"
                    footer_text = "_Cette vérification des faits a été effectuée automatiquement en utilisant plusieurs sources et l'analyse IA._"
                elif detected_language == "es":
                    fact_check_title = "🔍 **Resultados de verificación de hechos**"
                    your_message_label = "📝 **Su mensaje:**"
                    overall_verdict_label = "📊 **Veredicto general:**"
                    confidence_label = "📊 **Confianza:**"
                    checked_at_label = "⏰ Verificado en:"
                    footer_text = "_Esta verificación de hechos se realizó automáticamente utilizando múltiples fuentes y análisis de IA._"
                else:  # English
                    fact_check_title = "🔍 **Fact-Check Results**"
                    your_message_label = "📝 **Your Message:**"
                    overall_verdict_label = "📊 **Overall Verdict:**"
                    confidence_label = "📊 **Confidence:**"
                    checked_at_label = "⏰ Checked at:"
                    footer_text = "_This fact-check was performed automatically using multiple sources and AI analysis._"
                
                # Format the message
                message = f"""{fact_check_title}

{your_message_label} "{original_text[:100]}{'...' if len(original_text) > 100 else ''}"

{verdict_info['emoji']} {overall_verdict_label} {verdict_info['description']}
{confidence_label} {overall_confidence:.1%}
{language_note}
{detailed_analysis}

{checked_at_label} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{footer_text}"""
                
                logging.info(f"Formatted notification message in {detected_language}, length: {len(message)} chars")
                
            else:
                # Handle error case
                logging.warning(f"Fact-check was not successful, formatting error notification")
                error_message = fact_check_results.get('error', 'Unknown error occurred')
                logging.info(f"Error message: {error_message}")
                
                # Language-specific error text
                if detected_language == "ar":
                    fact_check_title = "🔍 **نتائج التحقق من الحقائق**"
                    your_message_label = "📝 **رسالتك:**"
                    status_label = "⚠️ **الحالة:** غير قادر على التحقق"
                    reason_label = "❌ **السبب:**"
                    attempted_at_label = "⏰ تمت المحاولة في:"
                    footer_text = "_يرجى المحاولة مرة أخرى لاحقاً أو الاتصال بالدعم إذا استمرت هذه المشكلة._"
                elif detected_language == "fr":
                    fact_check_title = "🔍 **Résultats de vérification des faits**"
                    your_message_label = "📝 **Votre message :**"
                    status_label = "⚠️ **Statut :** Impossible de vérifier"
                    reason_label = "❌ **Raison :**"
                    attempted_at_label = "⏰ Tenté à :"
                    footer_text = "_Veuillez réessayer plus tard ou contacter le support si ce problème persiste._"
                elif detected_language == "es":
                    fact_check_title = "🔍 **Resultados de verificación de hechos**"
                    your_message_label = "📝 **Su mensaje:**"
                    status_label = "⚠️ **Estado:** No se pudo verificar"
                    reason_label = "❌ **Razón:**"
                    attempted_at_label = "⏰ Intentado en:"
                    footer_text = "_Por favor, inténtelo de nuevo más tarde o contacte al soporte si este problema persiste._"
                else:  # English
                    fact_check_title = "🔍 **Fact-Check Results**"
                    your_message_label = "📝 **Your Message:**"
                    status_label = "⚠️ **Status:** Unable to verify"
                    reason_label = "❌ **Reason:**"
                    attempted_at_label = "⏰ Attempted at:"
                    footer_text = "_Please try again later or contact support if this issue persists._"
                
                message = f"""{fact_check_title}

{your_message_label} "{original_text[:100]}{'...' if len(original_text) > 100 else ''}"

{status_label}
{reason_label} {error_message}

{attempted_at_label} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{footer_text}"""
            
            return message
            
        except Exception as e:
            logging.error(f"Failed to format notification message: {e}")
            logging.error(f"Fact-check results structure: {fact_check_results}")
            logging.error(f"Exception traceback: {traceback.format_exc()}")
            # Fallback simple message
            return f"🔍 Fact-check complete for your message from {submission_timestamp}. Please check the results in your dashboard."
    
    def _get_verdict_info(self, verdict: str, confidence: float, language: str = "en") -> Dict[str, str]:
        """
        Get emoji and description for verdict
        
        Args:
            verdict: The overall verdict
            confidence: Confidence score (0-1)
            language: ISO language code
            
        Returns:
            Dict with emoji and description
        """
        # Language-specific verdict descriptions
        if language == "ar":
            verdict_mapping = {
                'SUPPORTED': {
                    'emoji': '✅' if confidence >= 0.8 else '🟢',
                    'description': 'صحيح على الأرجح' if confidence >= 0.8 else 'ربما صحيح'
                },
                'NOT_SUPPORTED': {
                    'emoji': '❌' if confidence >= 0.8 else '🔴',
                    'description': 'خاطئ على الأرجح' if confidence >= 0.8 else 'ربما خاطئ'
                },
                'MIXED': {
                    'emoji': '🟡',
                    'description': 'مختلط - بعض الأجزاء صحيحة وبعضها خاطئ'
                },
                'INSUFFICIENT_INFO': {
                    'emoji': '🟤',
                    'description': 'غير مؤكد - معلومات غير كافية'
                },
                'UNVERIFIED': {
                    'emoji': '⚪',
                    'description': 'غير مؤكد - لا يمكن التحقق'
                }
            }
        elif language == "fr":
            verdict_mapping = {
                'SUPPORTED': {
                    'emoji': '✅' if confidence >= 0.8 else '🟢',
                    'description': 'Probablement VRAI' if confidence >= 0.8 else 'Possiblement VRAI'
                },
                'NOT_SUPPORTED': {
                    'emoji': '❌' if confidence >= 0.8 else '🔴',
                    'description': 'Probablement FAUX' if confidence >= 0.8 else 'Possiblement FAUX'
                },
                'MIXED': {
                    'emoji': '🟡',
                    'description': 'MIXTE - Certaines parties vraies, d\'autres fausses'
                },
                'INSUFFICIENT_INFO': {
                    'emoji': '🟤',
                    'description': 'NON VÉRIFIÉ - Informations insuffisantes'
                },
                'UNVERIFIED': {
                    'emoji': '⚪',
                    'description': 'NON VÉRIFIÉ - Impossible de vérifier'
                }
            }
        elif language == "es":
            verdict_mapping = {
                'SUPPORTED': {
                    'emoji': '✅' if confidence >= 0.8 else '🟢',
                    'description': 'Probablemente VERDADERO' if confidence >= 0.8 else 'Posiblemente VERDADERO'
                },
                'NOT_SUPPORTED': {
                    'emoji': '❌' if confidence >= 0.8 else '🔴',
                    'description': 'Probablemente FALSO' if confidence >= 0.8 else 'Posiblemente FALSO'
                },
                'MIXED': {
                    'emoji': '🟡',
                    'description': 'MIXTO - Algunas partes verdaderas, algunas falsas'
                },
                'INSUFFICIENT_INFO': {
                    'emoji': '🟤',
                    'description': 'NO VERIFICADO - Información insuficiente'
                },
                'UNVERIFIED': {
                    'emoji': '⚪',
                    'description': 'NO VERIFICADO - No se pudo verificar'
                }
            }
        else:  # English (default)
            verdict_mapping = {
                'SUPPORTED': {
                    'emoji': '✅' if confidence >= 0.8 else '🟢',
                    'description': 'Likely TRUE' if confidence >= 0.8 else 'Possibly TRUE'
                },
                'NOT_SUPPORTED': {
                    'emoji': '❌' if confidence >= 0.8 else '🔴',
                    'description': 'Likely FALSE' if confidence >= 0.8 else 'Possibly FALSE'
                },
                'MIXED': {
                    'emoji': '🟡',
                    'description': 'MIXED - Some parts true, some false'
                },
                'INSUFFICIENT_INFO': {
                    'emoji': '🟤',
                    'description': 'UNVERIFIED - Insufficient information'
                },
                'UNVERIFIED': {
                    'emoji': '⚪',
                    'description': 'UNVERIFIED - Could not verify'
                }
            }
        
        return verdict_mapping.get(verdict, {
            'emoji': '❓',
            'description': 'UNKNOWN - Unexpected result' if language == "en" else 
                         'غير معروف - نتيجة غير متوقعة' if language == "ar" else
                         'INCONNU - Résultat inattendu' if language == "fr" else
                         'DESCONOCIDO - Resultado inesperado'
        })
    
    def _send_whatsapp_message(self, sender_phone: str, chat_jid: str, message: str) -> bool:
        """
        Send WhatsApp message using the WhatsApp API server
        
        Args:
            sender_phone: Phone number of original sender
            chat_jid: JID of the chat
            message: Message to send
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            # WhatsApp API server configuration
            WHATSAPP_API_URL = "http://localhost:9090"
            
            # Prepare the request
            recipient = chat_jid if chat_jid else sender_phone
            payload = {
                "recipient": recipient,
                "message": message
            }
            
            # Send request to WhatsApp API server
            response = requests.post(
                f"{WHATSAPP_API_URL}/send-message",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    logging.info(f"WhatsApp message sent successfully to {recipient}")
                    return True
                else:
                    logging.error(f"WhatsApp API returned error: {result.get('error', 'Unknown error')}")
                    return False
            else:
                logging.error(f"WhatsApp API server error: {response.status_code} - {response.text}")
                # Fallback to logging for manual sending
                self._log_notification_for_manual_send(sender_phone, chat_jid, message)
                return False
                
        except requests.exceptions.ConnectionError:
            logging.error("WhatsApp API server not available - logging for manual sending")
            self._log_notification_for_manual_send(sender_phone, chat_jid, message)
            return False
        except Exception as e:
            logging.error(f"Failed to send WhatsApp message: {e}")
            # Log the notification for manual sending
            self._log_notification_for_manual_send(sender_phone, chat_jid, message)
            return False
    
    def _log_notification_for_manual_send(self, sender_phone: str, chat_jid: str, message: str):
        """
        Log notification details for manual sending when automatic sending fails
        
        Args:
            sender_phone: Phone number of original sender
            chat_jid: JID of the chat
            message: Message to send
        """
        try:
            # Create a notifications log file
            log_file = "notifications_pending.txt"
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"""
=== Fact-Check Notification Pending ===
Timestamp: {datetime.now().isoformat()}
Sender Phone: {sender_phone}
Chat JID: {chat_jid}
Message:
{message}
=====================================

""")
            
            logging.info(f"Notification logged for manual sending to {sender_phone}")
            
        except Exception as e:
            logging.error(f"Failed to log notification for manual send: {e}")

    def _format_detailed_analysis(self, claim_results: List[Dict[str, Any]], 
                                overall_stats: Dict[str, Any], language: str = "en") -> str:
        """
        Format detailed analysis with individual claims, verdicts, and source links
        
        Args:
            claim_results: List of individual claim analysis results
            overall_stats: Overall statistics and verdict information
            language: ISO language code
            
        Returns:
            Formatted detailed analysis string
        """
        try:
            if not claim_results:
                # Language-specific "no claims" message
                if language == "ar":
                    return "📊 **التحليل:** لم يتم استخراج ادعاءات محددة للتحقق منها."
                elif language == "fr":
                    return "📊 **Analyse :** Aucune allégation spécifique n'a pu être extraite pour vérification."
                elif language == "es":
                    return "📊 **Análisis:** No se pudieron extraer afirmaciones específicas para verificación."
                else:  # English
                    return "📊 **Analysis:** No specific claims could be extracted for verification."
            
            # Get claim breakdown
            claim_breakdown = overall_stats.get('claim_breakdown', {})
            total_claims = claim_breakdown.get('total_claims', len(claim_results))
            true_claims = claim_breakdown.get('true_claims', 0)
            false_claims = claim_breakdown.get('false_claims', 0)
            partly_true_claims = claim_breakdown.get('partly_true_claims', 0)
            unverified_claims = claim_breakdown.get('unverified_claims', 0)
            
            # Start with claim summary
            analysis_parts = []
            
            # Language-specific labels
            if language == "ar":
                claims_analysis_label = f"📊 **تحليل الادعاءات ({total_claims} ادعاءات):**"
                true_label = "صحيح"
                false_label = "خاطئ"
                partly_true_label = "صحيح جزئياً"
                unverified_label = "غير مؤكد"
                key_sources_label = "🔗 **المصادر الرئيسية المرجعية:**"
                overall_analysis_label = "💡 **التحليل الشامل:**"
                more_claims_text = f"... و {len(claim_results) - 2} ادعاءات أخرى تم تحليلها"
            elif language == "fr":
                claims_analysis_label = f"📊 **Analyse des allégations ({total_claims} allégations) :**"
                true_label = "VRAI"
                false_label = "FAUX"
                partly_true_label = "PARTIELLEMENT VRAI"
                unverified_label = "NON VÉRIFIÉ"
                key_sources_label = "🔗 **Sources clés référencées :**"
                overall_analysis_label = "💡 **Analyse globale :**"
                more_claims_text = f"... et {len(claim_results) - 2} autres allégations analysées"
            elif language == "es":
                claims_analysis_label = f"📊 **Análisis de afirmaciones ({total_claims} afirmaciones):**"
                true_label = "VERDADERO"
                false_label = "FALSO"
                partly_true_label = "PARCIALMENTE VERDADERO"
                unverified_label = "NO VERIFICADO"
                key_sources_label = "🔗 **Fuentes clave referenciadas:**"
                overall_analysis_label = "💡 **Análisis general:**"
                more_claims_text = f"... y {len(claim_results) - 2} afirmaciones más analizadas"
            else:  # English
                claims_analysis_label = f"📊 **Claims Analysis ({total_claims} claims):**"
                true_label = "TRUE"
                false_label = "FALSE"
                partly_true_label = "PARTLY TRUE"
                unverified_label = "UNVERIFIED"
                key_sources_label = "🔗 **Key Sources Referenced:**"
                overall_analysis_label = "💡 **Overall Analysis:**"
                more_claims_text = f"... and {len(claim_results) - 2} more claims analyzed"
            
            # Add claim breakdown summary
            breakdown_text = claims_analysis_label
            if true_claims > 0:
                breakdown_text += f"\n✅ {true_claims} {true_label}"
            if false_claims > 0:
                breakdown_text += f"\n❌ {false_claims} {false_label}"
            if partly_true_claims > 0:
                breakdown_text += f"\n🟡 {partly_true_claims} {partly_true_label}"
            if unverified_claims > 0:
                breakdown_text += f"\n⚪ {unverified_claims} {unverified_label}"
            
            analysis_parts.append(breakdown_text)
            
            # Add individual claims with justifications and sources
            if len(claim_results) <= 3:  # Show all claims if 3 or fewer
                for i, result in enumerate(claim_results, 1):
                    claim_analysis = self._format_single_claim_analysis(i, result, language)
                    analysis_parts.append(claim_analysis)
            else:  # Show top claims for more than 3
                # Sort by confidence and show top 2
                sorted_results = sorted(claim_results, key=lambda x: x.get('confidence', 0), reverse=True)
                for i, result in enumerate(sorted_results[:2], 1):
                    claim_analysis = self._format_single_claim_analysis(i, result, language)
                    analysis_parts.append(claim_analysis)
                analysis_parts.append(more_claims_text)
            
            # Add key sources section
            key_sources = overall_stats.get('key_sources_referenced', [])
            if key_sources:
                sources_text = key_sources_label
                for i, source in enumerate(key_sources[:3], 1):  # Show top 3 sources
                    sources_text += f"\n{i}. {source}"
                analysis_parts.append(sources_text)
            
            # Add overall reasoning
            overall_reasoning = overall_stats.get('overall_reasoning', '')
            if overall_reasoning and len(overall_reasoning) > 50:
                reasoning_preview = overall_reasoning[:300] + "..." if len(overall_reasoning) > 300 else overall_reasoning
                
                # Add translation note if reasoning is in English but target language is different
                if language != "en" and self._is_english_text(reasoning_preview):
                    # Add a note that this is automatically translated for now
                    if language == "ar":
                        translation_note = " (مترجم تلقائياً)"
                    elif language == "fr":
                        translation_note = " (traduit automatiquement)"
                    elif language == "es":
                        translation_note = " (traducido automáticamente)"
                    else:
                        translation_note = ""
                    reasoning_preview += translation_note
                
                analysis_parts.append(f"{overall_analysis_label} {reasoning_preview}")
            
            return "\n\n".join(analysis_parts)
            
        except Exception as e:
            logging.error(f"Failed to format detailed analysis: {e}")
            if language == "ar":
                fallback = f"📊 **التحليل:** {overall_stats.get('overall_reasoning', 'تم إكمال التحليل مع بعض المشاكل التقنية.')}"
            elif language == "fr":
                fallback = f"📊 **Analyse :** {overall_stats.get('overall_reasoning', 'Analyse terminée avec quelques problèmes techniques.')}"
            elif language == "es":
                fallback = f"📊 **Análisis:** {overall_stats.get('overall_reasoning', 'Análisis completado con algunos problemas técnicos.')}"
            else:
                fallback = f"📊 **Analysis:** {overall_stats.get('overall_reasoning', 'Analysis completed with some technical issues.')}"
            return fallback
    
    def _format_single_claim_analysis(self, claim_num: int, result: Dict[str, Any], language: str = "en") -> str:
        """
        Format analysis for a single claim with verdict and sources
        
        Args:
            claim_num: Claim number for display
            result: Individual claim result dictionary
            language: ISO language code
            
        Returns:
            Formatted claim analysis string
        """
        try:
            claim = result.get('claim', 'Unknown claim')
            verdict = result.get('verdict', 'UNVERIFIED')
            confidence = result.get('confidence', 0.0)
            justification = result.get('justification', 'No justification provided')
            supporting_sources = result.get('supporting_sources', [])
            contradicting_sources = result.get('contradicting_sources', [])
            
            # Get verdict emoji
            verdict_emojis = {
                'TRUE': '✅',
                'FALSE': '❌', 
                'PARTLY_TRUE': '🟡',
                'UNVERIFIED': '⚪'
            }
            emoji = verdict_emojis.get(verdict, '❓')
            
            # Format claim text (truncate if too long)
            claim_text = claim[:80] + "..." if len(claim) > 80 else claim
            
            # Language-specific labels
            if language == "ar":
                claim_label = f"**الادعاء {claim_num}:**"
                # Translate verdict to Arabic
                verdict_translations = {
                    'TRUE': 'صحيح',
                    'FALSE': 'خاطئ',
                    'PARTLY_TRUE': 'صحيح جزئياً',
                    'UNVERIFIED': 'غير مؤكد'
                }
                verdict_display = verdict_translations.get(verdict, verdict)
            elif language == "fr":
                claim_label = f"**Allégation {claim_num} :**"
                verdict_translations = {
                    'TRUE': 'VRAI',
                    'FALSE': 'FAUX',
                    'PARTLY_TRUE': 'PARTIELLEMENT VRAI',
                    'UNVERIFIED': 'NON VÉRIFIÉ'
                }
                verdict_display = verdict_translations.get(verdict, verdict)
            elif language == "es":
                claim_label = f"**Afirmación {claim_num}:**"
                verdict_translations = {
                    'TRUE': 'VERDADERO',
                    'FALSE': 'FALSO',
                    'PARTLY_TRUE': 'PARCIALMENTE VERDADERO',
                    'UNVERIFIED': 'NO VERIFICADO'
                }
                verdict_display = verdict_translations.get(verdict, verdict)
            else:  # English
                claim_label = f"**Claim {claim_num}:**"
                verdict_display = verdict
            
            # Start building the analysis
            analysis = f"{claim_label} {claim_text}\n{emoji} **{verdict_display}** ({confidence:.0%})"
            
            # Add justification (first sentence or up to 150 chars)
            if justification and len(justification) > 20:
                justification_preview = justification.split('.')[0] + '.'
                if len(justification_preview) > 150:
                    justification_preview = justification[:150] + "..."
                
                # If the justification is in English but we want another language, provide a fallback
                if language != "en" and self._is_english_text(justification_preview):
                    # Add a note that this is automatically translated for now
                    if language == "ar":
                        justification_label = "📋 التبرير:"
                        translation_note = " (مترجم تلقائياً)"
                    elif language == "fr":
                        justification_label = "📋 Justification :"
                        translation_note = " (traduit automatiquement)"
                    elif language == "es":
                        justification_label = "📋 Justificación:"
                        translation_note = " (traducido automáticamente)"
                    else:
                        justification_label = "📋"
                        translation_note = ""
                    
                    analysis += f"\n{justification_label} {justification_preview}{translation_note}"
                else:
                    # Add language-specific justification label
                    if language == "ar":
                        justification_label = "📋 التبرير:"
                    elif language == "fr":
                        justification_label = "📋 Justification :"
                    elif language == "es":
                        justification_label = "📋 Justificación:"
                    else:
                        justification_label = "📋"
                    
                    analysis += f"\n{justification_label} {justification_preview}"
            
            # Add key sources that support or contradict
            sources_added = 0
            if supporting_sources and sources_added < 2:
                for source in supporting_sources[:1]:  # Show 1 supporting source
                    if source and source.startswith('http'):
                        analysis += f"\n✅ {source}"
                        sources_added += 1
                        
            if contradicting_sources and sources_added < 2:
                for source in contradicting_sources[:1]:  # Show 1 contradicting source
                    if source and source.startswith('http'):
                        analysis += f"\n❌ {source}"
                        sources_added += 1
            
            return analysis
            
        except Exception as e:
            logging.error(f"Failed to format single claim analysis: {e}")
            # Language-specific error message
            if language == "ar":
                error_msg = f"**الادعاء {claim_num}:** حدث خطأ في التحليل"
            elif language == "fr":
                error_msg = f"**Allégation {claim_num} :** Erreur d'analyse survenue"
            elif language == "es":
                error_msg = f"**Afirmación {claim_num}:** Ocurrió un error de análisis"
            else:
                error_msg = f"**Claim {claim_num}:** Analysis error occurred"
            return error_msg
        
    def _is_english_text(self, text: str) -> bool:
        """
        Simple heuristic to detect if text is primarily in English
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be in English
        """
        if not text:
            return True
            
        # Simple heuristic: check for common English words and patterns
        english_indicators = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
            'according', 'sources', 'evidence', 'information', 'claim', 'statement'
        ]
        
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_indicators if word in text_lower)
        
        # If we find several English indicator words, likely English
        return english_word_count >= 3

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rabbitmq_workers.py <worker_type>")
        print("Worker types: image, video, pdf, text, fact_check")
        sys.exit(1)
    
    worker_type = sys.argv[1].lower()
    
    workers = {
        'image': ImageProcessingWorker,
        'video': VideoProcessingWorker,
        'pdf': PDFProcessingWorker,
        'text': TextProcessingWorker,
        'fact_check': FactCheckingWorker
    }
    
    if worker_type not in workers:
        print(f"Invalid worker type: {worker_type}")
        print(f"Available types: {', '.join(workers.keys())}")
        sys.exit(1)
    
    # Start the worker
    worker = workers[worker_type]()
    logging.info(f"Starting {worker_type} worker...")
    worker.start_consuming()
