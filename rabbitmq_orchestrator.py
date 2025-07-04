"""
RabbitMQ Orchestrator for WhatsApp/Telegram Message Processing Pipeline

This module handles:
1. Receiving incoming messages
2. Storing messages in the database 
3. Routing messages to appropriate RabbitMQ queues based on content type
4. Managing worker queues for different processing types
"""

import os
import json
import sqlite3
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import pika
from pika.adapters.blocking_connection import BlockingChannel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RabbitMQ Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

# Queue Names
IMAGE_QUEUE = "image_processing_queue"
VIDEO_QUEUE = "video_processing_queue"  
PDF_QUEUE = "pdf_processing_queue"
TEXT_QUEUE = "text_processing_queue"
FACT_CHECK_QUEUE = "fact_check_queue"
RESULTS_QUEUE = "processing_results_queue"

# Database Configuration
DB_PATH = os.getenv("DB_PATH", "databases/raw_data.db")

class MessageRecord(BaseModel):
    """Pydantic model for incoming message records"""
    source_type: str  # 'whatsapp' or 'telegram'
    sender_phone: str
    is_group_message: bool = False
    group_name: Optional[str] = None
    channel_name: Optional[str] = None
    chat_jid: str
    content_type: str  # 'audio', 'video', 'pdf', 'image', 'text', 'document', 'link'
    content_url: Optional[str] = None
    raw_text: Optional[str] = None
    user_identifier: str
    priority: str = "normal"
    metadata: Optional[Dict[str, Any]] = None

class RabbitMQOrchestrator:
    """
    Main orchestrator class that manages RabbitMQ connections and message routing
    """
    
    def __init__(self):
        """Initialize the orchestrator with RabbitMQ connection and database"""
        self.connection = None
        self.channel = None
        self.setup_rabbitmq()
        self.setup_database()
        
    def setup_rabbitmq(self):
        """Establish RabbitMQ connection and declare queues"""
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare all processing queues
            queues = [IMAGE_QUEUE, VIDEO_QUEUE, PDF_QUEUE, TEXT_QUEUE, RESULTS_QUEUE]
            for queue in queues:
                self.channel.queue_declare(queue=queue, durable=True)
                logger.info(f"Declared queue: {queue}")
                
            logger.info("RabbitMQ connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup RabbitMQ: {e}")
            raise
    
    def setup_database(self):
        """Initialize database and create tables if they don't exist"""
        try:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            
            with sqlite3.connect(DB_PATH) as conn:
                # Read and execute schema
                schema_path = "schema.sql"
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schema = f.read()
                    conn.executescript(schema)
                else:
                    # Create table directly if schema file not found
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS raw_data (
                            ID TEXT PRIMARY KEY,
                            UUID TEXT NOT NULL,
                            source_type TEXT NOT NULL CHECK (source_type IN ('whatsapp', 'telegram')),
                            sender_phone TEXT NOT NULL,
                            is_group_message BOOLEAN NOT NULL DEFAULT 0,
                            group_name TEXT,
                            channel_name TEXT,
                            chat_jid TEXT,
                            content_type TEXT NOT NULL CHECK (content_type IN ('audio', 'video', 'pdf', 'image', 'text', 'document', 'link')),
                            content_url TEXT,
                            raw_text TEXT,
                            submission_timestamp DATETIME NOT NULL,
                            processing_status TEXT NOT NULL DEFAULT 'pending',
                            user_identifier TEXT NOT NULL,
                            priority TEXT NOT NULL DEFAULT 'normal',
                            metadata TEXT
                        )
                    """)
                    
                    # Create indexes
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_raw_data_timestamp ON raw_data(submission_timestamp)",
                        "CREATE INDEX IF NOT EXISTS idx_raw_data_user ON raw_data(user_identifier)",
                        "CREATE INDEX IF NOT EXISTS idx_raw_data_status ON raw_data(processing_status)",
                        "CREATE INDEX IF NOT EXISTS idx_raw_data_sender_phone ON raw_data(sender_phone)",
                        "CREATE INDEX IF NOT EXISTS idx_raw_data_group ON raw_data(is_group_message)"
                    ]
                    
                    for index in indexes:
                        conn.execute(index)
                        
                conn.commit()
                logger.info("Database setup completed successfully")
                
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
    
    def store_message(self, message: MessageRecord) -> str:
        """
        Store incoming message in the database
        
        Args:
            message: MessageRecord object containing message data
            
        Returns:
            str: Generated message ID
        """
        try:
            # Generate unique identifiers
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
            message_uuid = str(uuid.uuid4())
            
            # Prepare metadata as JSON string
            metadata_json = json.dumps(message.metadata) if message.metadata else None
            
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO raw_data (
                        ID, UUID, source_type, sender_phone, is_group_message,
                        group_name, channel_name, chat_jid, content_type, content_url,
                        raw_text, submission_timestamp, processing_status,
                        user_identifier, priority, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message_id, message_uuid, message.source_type, message.sender_phone,
                    message.is_group_message, message.group_name, message.channel_name,
                    message.chat_jid, message.content_type, message.content_url,
                    message.raw_text, datetime.now().isoformat(), 'pending',
                    message.user_identifier, message.priority, metadata_json                ))
                conn.commit()                
            logger.info(f"Message stored successfully with ID: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            raise
    
    def get_processing_queue(self, content_type: str) -> str:
        """
        Determine which RabbitMQ queue to route the message to based on content type
        
        Args:
            content_type: The type of content (image, video, pdf, text, etc.)
            
        Returns:
            str: Queue name for the content type
        """
        queue_mapping = {
            'image': IMAGE_QUEUE,
            'video': VIDEO_QUEUE,
            'audio': VIDEO_QUEUE,  # Audio uses video transcription queue
            'pdf': PDF_QUEUE,
            'document': PDF_QUEUE,  # Documents treated as PDFs
            'text': TEXT_QUEUE,
            'link': TEXT_QUEUE  # Links processed with text queue
        }
        
        return queue_mapping.get(content_type.lower(), TEXT_QUEUE)
    
    def route_to_queue(self, message_id: str, queue_name: str):
        """
        Route a message to the appropriate RabbitMQ queue for processing
        
        Args:
            message_id: ID of the message to process (using ID column)
            queue_name: Name of the queue to route to
        """
        try:
            # Retrieve full message data from database using ID column
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM raw_data WHERE ID = ?
                """, (message_id,))
                row = cursor.fetchone()
                
                if not row:
                    raise ValueError(f"Message with ID {message_id} not found")
                
                # Convert row to dictionary
                message_data = dict(row)
                
                # Parse metadata back from JSON
                if message_data['metadata']:
                    message_data['metadata'] = json.loads(message_data['metadata'])
                
            # Publish message to appropriate queue
            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message_data),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    priority=1 if message_data.get('priority') == 'high' else 0
                )
            )
            
            # Update processing status
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    UPDATE raw_data 
                    SET processing_status = 'queued'
                    WHERE ID = ?
                """, (message_id,))
                conn.commit()
            
            logger.info(f"Message {message_id} routed to queue: {queue_name}")
            
        except Exception as e:
            logger.error(f"Failed to route message {message_id} to queue {queue_name}: {e}")
            # Update status to failed
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("""
                        UPDATE raw_data 
                        SET processing_status = 'failed'
                        WHERE ID = ?
                    """, (message_id,))
                    conn.commit()
            except:
                pass
            raise
    
    def process_message(self, message: MessageRecord) -> str:
        """
        Complete message processing workflow: store and route to queue
        
        Args:
            message: MessageRecord object containing message data
            
        Returns:
            str: Generated message ID
        """
        try:
            # Step 1: Store message in database
            message_id = self.store_message(message)
            
            # Step 2: Determine appropriate queue
            queue_name = self.get_processing_queue(message.content_type)
            
            # Step 3: Route to queue
            self.route_to_queue(message_id, queue_name)
            
            logger.info(f"Message {message_id} processed successfully, routed to {queue_name}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise
    
    def close_connection(self):
        """Close RabbitMQ connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("RabbitMQ connection closed")

# FastAPI application for message ingestion
app = FastAPI(
    title="RabbitMQ Message Orchestrator",
    description="Orchestrates message processing through RabbitMQ queues",
    version="1.0.0"
)

# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator
    orchestrator = RabbitMQOrchestrator()
    logger.info("RabbitMQ Orchestrator started successfully")

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean up on shutdown"""
    global orchestrator
    if orchestrator:
        orchestrator.close_connection()
    logger.info("RabbitMQ Orchestrator shutdown completed")

@app.post("/process-message")
async def process_message_endpoint(message: MessageRecord):
    """
    API endpoint to receive and process messages
    
    Args:
        message: MessageRecord containing message data
    
    Returns:
        dict: Success response with message ID
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
        message_id = orchestrator.process_message(message)
        
        return {
            "success": True,
            "message_id": message_id,
            "status": "queued_for_processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rabbitmq_connected": orchestrator.connection and not orchestrator.connection.is_closed if orchestrator else False
    }

@app.get("/queues/status")
async def queue_status():
    """Get status of all processing queues"""
    try:
        if not orchestrator or not orchestrator.channel:
            raise HTTPException(status_code=500, detail="RabbitMQ not connected")
        
        queues = [IMAGE_QUEUE, VIDEO_QUEUE, PDF_QUEUE, TEXT_QUEUE, FACT_CHECK_QUEUE, RESULTS_QUEUE]
        status = {}
        
        for queue in queues:
            try:
                method = orchestrator.channel.queue_declare(queue=queue, passive=True)
                status[queue] = {
                    "message_count": method.method.message_count,
                    "consumer_count": method.method.consumer_count
                }
            except Exception as e:
                status[queue] = {"error": str(e)}
        
        return {"queues": status}
        
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
