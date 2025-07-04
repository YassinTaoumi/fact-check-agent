"""
Results Consumer for RabbitMQ Pipeline

This module consumes results from the processing_results_queue and triggers
fact-checking for completed messages that have text content available.
"""
import sys
import os
import json
import sqlite3
import logging
import asyncio
from typing import Dict, Any, Optional
import pika
from datetime import datetime
import traceback

# Add fact-checker to path
fact_checker_path = os.path.join(os.path.dirname(__file__), 'fact-checker')
if fact_checker_path not in sys.path:
    import sys
    sys.path.append(fact_checker_path)

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

class ResultsConsumer:
    """Consumer that monitors processing results and triggers fact-checking"""
    
    def __init__(self):
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
            
            # Declare the queues
            self.channel.queue_declare(queue=RESULTS_QUEUE, durable=True)
            self.channel.queue_declare(queue=FACT_CHECK_QUEUE, durable=True)
            
            # Set quality of service
            self.channel.basic_qos(prefetch_count=1)
            
            logger.info("ResultsConsumer connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup RabbitMQ for ResultsConsumer: {e}")
            raise
    
    def should_fact_check(self, message_id: str) -> bool:
        """
        Check if a message should be fact-checked based on available processed data
        
        Args:
            message_id: ID of the message to check
            
        Returns:
            bool: True if fact-checking should be triggered
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                
                # Check if there's already a fact-check result
                cursor.execute("""
                    SELECT COUNT(*) FROM fact_check_results WHERE message_id = ?
                """, (message_id,))
                
                if cursor.fetchone()[0] > 0:
                    logger.info(f"Message {message_id} already fact-checked, skipping")
                    return False
                
                # Check if there's text content available in processed results
                cursor.execute("""
                    SELECT 
                        cleaned_text, ocr_text, video_transcription, 
                        pdf_text_extraction, link_content
                    FROM processed 
                    WHERE message_id = ? AND processing_status = 'completed'
                """, (message_id,))
                
                row = cursor.fetchone()
                if not row:
                    # Check if there's raw text in the original message
                    cursor.execute("""
                        SELECT raw_text, content_type 
                        FROM raw_data 
                        WHERE ID = ? AND raw_text IS NOT NULL AND raw_text != ''
                    """, (message_id,))
                    
                    raw_row = cursor.fetchone()
                    if raw_row and raw_row[1] == 'text':
                        logger.info(f"Message {message_id} has raw text, will fact-check")
                        return True
                    
                    logger.info(f"Message {message_id} has no text content, skipping fact-check")
                    return False
                
                # Check if any text content is available
                cleaned_text, ocr_text, video_transcription, pdf_text_extraction, link_content = row
                
                text_sources = [
                    (cleaned_text, "cleaned_text"),
                    (pdf_text_extraction, "pdf_extraction"),
                    (video_transcription, "transcription"),
                    (ocr_text, "ocr"),
                    (link_content, "link_content")
                ]
                
                for text_content, source in text_sources:
                    if text_content:
                        # Try to extract text from JSON if needed
                        try:
                            if isinstance(text_content, str) and text_content.startswith('{'):
                                data = json.loads(text_content)
                                if isinstance(data, dict):
                                    # Look for common text fields
                                    for field in ['text', 'transcription', 'content', 'cleaned_text']:
                                        if field in data and data[field]:
                                            logger.info(f"Message {message_id} has {source} content, will fact-check")
                                            return True
                            else:
                                if text_content.strip():
                                    logger.info(f"Message {message_id} has {source} content, will fact-check")
                                    return True
                        except:
                            # If JSON parsing fails, check if it's plain text
                            if text_content.strip():
                                logger.info(f"Message {message_id} has {source} content, will fact-check")
                                return True
                
                logger.info(f"Message {message_id} has no meaningful text content, skipping fact-check")
                return False
                
        except Exception as e:
            logger.error(f"Failed to check if message {message_id} should be fact-checked: {e}")
            return False
    
    def queue_for_fact_checking(self, message_id: str):
        """
        Queue a message for fact-checking by sending it to the fact-check queue
        
        Args:
            message_id: ID of the message to fact-check
        """
        try:
            # Get the message data from database
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM raw_data WHERE ID = ?
                """, (message_id,))
                row = cursor.fetchone()
                
                if not row:
                    logger.error(f"Message {message_id} not found in database")
                    return
                
                message_data = dict(row)
                
                # Parse metadata back from JSON if it exists
                if message_data.get('metadata'):
                    try:
                        message_data['metadata'] = json.loads(message_data['metadata'])
                    except:
                        pass
            
            # Send to fact-check queue
            self.channel.basic_publish(
                exchange='',
                routing_key=FACT_CHECK_QUEUE,
                body=json.dumps(message_data),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    priority=0
                )
            )
            
            logger.info(f"Message {message_id} queued for fact-checking")
            
        except Exception as e:
            logger.error(f"Failed to queue message {message_id} for fact-checking: {e}")
    
    def callback(self, ch, method, properties, body):
        """
        RabbitMQ callback function to process results
        
        Args:
            ch: Channel object
            method: Method object
            properties: Properties object
            body: Message body
        """
        try:
            # Parse result data
            result_data = json.loads(body)
            message_id = result_data.get('message_id')
            worker_name = result_data.get('worker_name')
            result = result_data.get('result', {})
            
            logger.info(f"Received result from {worker_name} for message {message_id}")
            
            # Check if processing was successful
            if result.get('status') == 'success':
                # Check if this message should be fact-checked
                if self.should_fact_check(message_id):
                    self.queue_for_fact_checking(message_id)
                else:
                    logger.info(f"Message {message_id} does not need fact-checking")
            else:
                logger.info(f"Message {message_id} processing failed, skipping fact-check")
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"Failed to process result: {e}")
            logger.error(traceback.format_exc())
            
            # Acknowledge message to remove it from queue
            ch.basic_ack(delivery_tag=method.delivery_tag)
    
    def start_consuming(self):
        """Start consuming messages from the results queue"""
        logger.info("ResultsConsumer starting to consume from queue: processing_results_queue")
        
        self.channel.basic_consume(
            queue=RESULTS_QUEUE,
            on_message_callback=self.callback
        )
        
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("ResultsConsumer stopping...")
            self.channel.stop_consuming()
            self.connection.close()
    
    def close_connection(self):
        """Close RabbitMQ connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()

if __name__ == "__main__":
    consumer = ResultsConsumer()
    logger.info("Starting Results Consumer...")
    consumer.start_consuming()
