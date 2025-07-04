# processing_storage_api.py - Combined Processing and Storage API (FIXED)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum
import os
import mimetypes
import logging
import uuid
import aiofiles
import shutil
import sqlite3
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import socket

# Import RabbitMQ Orchestrator
from rabbitmq_orchestrator import RabbitMQOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RabbitMQ orchestrator
orchestrator = None

app = FastAPI(
    title="Content Processing & Storage API",
    description="API for processing, validating, and storing content files",
    version="1.0.0"
)

# Add request logging middleware (temporarily disabled for debugging)
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start_time = time.time()
#     logger.info(f"🌐 Incoming request: {request.method} {request.url}")
#     logger.info(f"🌐 Headers: {dict(request.headers)}")
#     if request.method == "POST":
#         # Try to read body for POST requests
#         try:
#             body = await request.body()
#             logger.info(f"🌐 Body (first 500 chars): {body[:500]}")
#         except:
#             logger.info("🌐 Could not read request body")
#     
#     response = await call_next(request)
#     process_time = time.time() - start_time
#     logger.info(f"🌐 Response: {response.status_code} (took {process_time:.3f}s)")
#     return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff"}
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv", "video/webm", "video/flv"}
SUPPORTED_AUDIO_TYPES = {"audio/ogg", "audio/mp3", "audio/wav", "audio/m4a", "audio/flac", "audio/aac"}
SUPPORTED_DOCUMENT_TYPES = {"application/pdf", "text/plain", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

# Directories
UPLOAD_DIRECTORY = "uploads"
PROCESSED_DIRECTORY = "processed"
FILE_STORAGE_PATH = "file_storage"
DB_PATH = "databases/raw_data.db"

# Ensure directories exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
os.makedirs(FILE_STORAGE_PATH, exist_ok=True)
os.makedirs("databases", exist_ok=True)

# Enums
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    FAILED = "failed"
    REJECTED = "rejected"

class ContentType(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    PDF = "pdf"
    DOCUMENT = "document"
    LINK = "link"

class SourceType(str, Enum):
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    
class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# Pydantic Models
class ProcessingRequest(BaseModel):
    message_id: str
    chat_jid: str
    chat_name: str
    sender_jid: str
    sender_name: str
    user_identifier: str
    content: str
    content_type: ContentType
    media_filename: Optional[str] = None
    media_size: Optional[int] = None
    media_path: Optional[str] = None
    is_from_me: bool = False
    is_group: bool = False
    timestamp: str
    source_type: SourceType = SourceType.WHATSAPP
    priority: Priority = Priority.NORMAL

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    stored_id: Optional[str] = None
    file_storage_path: Optional[str] = None
    processing_job_id: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    should_retry: bool = False

# Database Helper Functions
def init_database():
    """Initialize the database with the correct schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
      # Create raw_data table with proper schema
    cursor.execute('''
CREATE TABLE IF NOT EXISTS raw_data (
    ID TEXT PRIMARY KEY,
    UUID TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('whatsapp', 'telegram')),
    sender_phone TEXT,
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
''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_data_timestamp ON raw_data(submission_timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_data_user ON raw_data(user_identifier)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_data_status ON raw_data(processing_status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_data_sender_phone ON raw_data(sender_phone)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_data_group ON raw_data(is_group_message)')
    
    conn.commit()
    conn.close()
    
    logger.info("Database initialized successfully")

def generate_id(message_id: str, chat_jid: str, timestamp: str) -> str:
    """Generate unique ID for database record"""
    data = f"{message_id}_{chat_jid}_{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def generate_uuid() -> str:
    """Generate UUID for database record"""
    return str(uuid.uuid4()).replace("-", "")

def store_file_in_storage(source_path: str, user_identifier: str, filename: str) -> str:
    """Store file in organized file storage structure"""
    if not source_path or not os.path.exists(source_path):
        return ""
    
    # Create user directory in file storage
    user_dir = os.path.join(FILE_STORAGE_PATH, user_identifier)
    os.makedirs(user_dir, exist_ok=True)
    
    # Generate unique filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    unique_filename = f"{timestamp}_{name}{ext}"
    
    destination_path = os.path.join(user_dir, unique_filename)
    
    try:
        # Copy file to storage location
        shutil.copy2(source_path, destination_path)
        logger.info(f"File stored: {destination_path}")
        return destination_path
    except Exception as e:
        logger.error(f"Failed to store file {source_path} to {destination_path}: {e}")
        raise

def safe_convert_value(value, field_name="unknown"):
    """Safely convert values to SQLite-compatible types"""
    if value is None:
        return ""
    elif isinstance(value, bool):
        return int(value)  # Convert bool to int for SQLite
    elif isinstance(value, (int, float)):
        return value
    elif isinstance(value, str):
        return value
    elif isinstance(value, dict):
        logger.warning(f"Converting dict to JSON string for field {field_name}: {value}")
        return json.dumps(value)
    elif isinstance(value, list):
        logger.warning(f"Converting list to JSON string for field {field_name}: {value}")
        return json.dumps(value)
    else:
        logger.warning(f"Converting unknown type {type(value)} to string for field {field_name}: {value}")
        return str(value)

def store_in_raw_database(req, file_storage_path: str = None) -> str:
    """Store message data in raw database with proper type conversion"""    # Debug logging at the beginning
    logger.info(f"store_in_raw_database called with req type: {req}")
    logger.info(f"req attributes: {vars(req) if hasattr(req, '__dict__') else 'No __dict__'}")
    
    # Check all the key attributes we need
    logger.info(f"DEBUG: req.message_id = '{getattr(req, 'message_id', 'MISSING')}'")
    logger.info(f"DEBUG: req.chat_jid = '{getattr(req, 'chat_jid', 'MISSING')}'")
    logger.info(f"DEBUG: req.sender_jid = '{getattr(req, 'sender_jid', 'MISSING')}'")
    logger.info(f"DEBUG: req.user_identifier = '{getattr(req, 'user_identifier', 'MISSING')}'")
    
    record_id = generate_id(req.message_id, req.chat_jid, req.timestamp)
    record_uuid = generate_uuid()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()     
    try:
        user_identifier_full = getattr(req, 'user_identifier', '')      
        # Store the full sender_jid in sender_phone field
        sender_phone = getattr(req, 'sender_jid', '')
        logger.info(f"DEBUG: sender_phone (sender_jid) being inserted: {sender_phone}")
        logger.info(f"DEBUG: user_identifier_full being inserted: {user_identifier_full}")
        
        # Fallback to user_identifier if sender_jid is not available
        if not sender_phone:
            sender_phone = getattr(req, 'user_identifier', '')

        # Get chat_jid with explicit debug logging (fix: use chat_jid not sender_jid)
        chat_jid_value = getattr(req, 'chat_jid', '')
        logger.info(f"DEBUG: chat_jid extracted: '{chat_jid_value}' (type: {type(chat_jid_value)})")
        
        # Determine if it's a group or channel
        is_group_message = getattr(req, 'is_group', False)
        group_name = None
        channel_name = None

        if is_group_message:
            chat_name = getattr(req, 'chat_name', '')
            if '@newsletter' in chat_jid_value:
                channel_name = chat_name
                is_group_message = False # It's a channel, not a group
            else:
                group_name = chat_name        # Safely get metadata, defaulting to None if it doesn't exist
        metadata_value = getattr(req, 'metadata', None)
        metadata_json = json.dumps(metadata_value) if metadata_value else None
        
        # Determine content URL based on content type and file storage path
        content_url = None
        content_type_value = getattr(req, 'content_type', 'text')
        
        # For media files (image, video, audio), store the file storage path as content_url
        if hasattr(content_type_value, 'value'):
            content_type_str = content_type_value.value
        else:
            content_type_str = str(content_type_value)
            
        if content_type_str in ['image', 'video', 'audio', 'document', 'pdf'] and file_storage_path:
            content_url = file_storage_path
            logger.info(f"Setting content_url to file storage path for {content_type_str}: {file_storage_path}")
          # Build a dictionary of values to insert - convert empty strings to None
        def none_if_empty(value):
            """Convert empty strings to None for proper NULL storage in SQLite"""
            if value == '' or value is None:
                return None
            return value
        
        data_to_insert = {
            "ID": record_id,
            "UUID": record_uuid,
            "source_type": getattr(req, 'source_type', 'whatsapp').value if hasattr(getattr(req, 'source_type', 'whatsapp'), 'value') else str(getattr(req, 'source_type', 'whatsapp')),
            "sender_phone": none_if_empty(sender_phone),
            "is_group_message": int(is_group_message),
            "group_name": none_if_empty(group_name),
            "channel_name": none_if_empty(channel_name),
            "chat_jid": none_if_empty(chat_jid_value),
            "content_type": content_type_str,
            "content_url": none_if_empty(content_url),
            "raw_text": getattr(req, 'content', ''),
            "submission_timestamp": getattr(req, 'timestamp', datetime.now().isoformat()),
            "processing_status": "pending",            
            "user_identifier": none_if_empty(user_identifier_full),            "priority": getattr(req, 'priority', 'normal').value if hasattr(getattr(req, 'priority', 'normal'), 'value') else str(getattr(req, 'priority', 'normal')),
            "metadata": metadata_json
        }
        logger.info(f"DEBUG: Data to insert: {data_to_insert}")
        columns = ', '.join(data_to_insert.keys())
        placeholders = ', '.join('?' * len(data_to_insert))
        sql_statement = f'INSERT OR REPLACE INTO raw_data ({columns}) VALUES ({placeholders})'
        values = tuple(data_to_insert.values())
        logger.info(f"DEBUG: About to insert chat_jid: '{data_to_insert['chat_jid']}'")
        cursor.execute(sql_statement, values)
        conn.commit()
        
        logger.info(f"Successfully stored raw data with ID: {record_id}, UUID: {record_uuid}")
        logger.info(f"Chat JID: {chat_jid_value}")
        logger.info(f"Content Type: {content_type_str}")
        logger.info(f"Content URL: {content_url}")
        logger.info(f"Group/Channel info - is_group: {is_group_message}, group_name: {group_name}, channel_name: {channel_name}")
        return record_id  # Return the ID instead of UUID for orchestrator compatibility

    except Exception as e:
        logger.error(f"Error storing in database: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        conn.rollback()
        raise
    finally:
        conn.close()
# File Processing Functions
async def save_uploaded_file(file: UploadFile, user_id: str) -> str:
    """Save uploaded file to temporary location"""
    user_upload_dir = os.path.join(UPLOAD_DIRECTORY, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(user_upload_dir, filename)
    
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        return file_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

def validate_file(file_path: str, content_type: str = None) -> Dict[str, Any]:
    """Validate file based on type, size, format etc."""
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "detected_content_type": None,
        "file_size": 0,
        "detection_score": 1.0
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            result["valid"] = False
            result["errors"].append("File not found")
            return result
        
        # Check file size
        file_size = os.path.getsize(file_path)
        result["file_size"] = file_size
        
        if file_size > MAX_FILE_SIZE:
            result["valid"] = False
            result["errors"].append(f"File too large ({file_size} bytes). Max allowed size is {MAX_FILE_SIZE} bytes")
        
        # Always detect content type from file extension first (ignore application/octet-stream)
        file_ext = os.path.splitext(file_path)[1].lower()
        extension_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', 
            '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.mp4': 'video/mp4', '.avi': 'video/avi', '.mov': 'video/mov', 
            '.mkv': 'video/mkv', '.webm': 'video/webm', '.flv': 'video/flv',
            '.ogg': 'audio/ogg', '.mp3': 'audio/mp3', '.wav': 'audio/wav',
            '.m4a': 'audio/m4a', '.flac': 'audio/flac', '.aac': 'audio/aac',
            '.pdf': 'application/pdf', '.txt': 'text/plain',
            '.doc': 'application/msword', 
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        
        detected_content_type = extension_map.get(file_ext)
        
        if detected_content_type:
            logger.info(f"🔍 Detected content type from extension {file_ext}: {detected_content_type}")
        else:
            # Try mimetypes as fallback, but only if extension detection failed
            detected_content_type, _ = mimetypes.guess_type(file_path)
            if detected_content_type and detected_content_type != "application/octet-stream":
                logger.info(f"🔍 Detected content type from mimetypes: {detected_content_type}")
            else:
                logger.warning(f"🔍 Could not detect content type for file with extension: {file_ext}")
                detected_content_type = None
        
        result["detected_content_type"] = detected_content_type
        
        # Validate based on detected content type
        if detected_content_type:
            if detected_content_type.startswith("image/"):
                if detected_content_type not in SUPPORTED_IMAGE_TYPES:
                    result["valid"] = False
                    result["errors"].append(f"Unsupported image format: {detected_content_type}")
            elif detected_content_type.startswith("video/"):
                if detected_content_type not in SUPPORTED_VIDEO_TYPES:
                    result["valid"] = False
                    result["errors"].append(f"Unsupported video format: {detected_content_type}")
            elif detected_content_type.startswith("audio/"):
                if detected_content_type not in SUPPORTED_AUDIO_TYPES:
                    result["valid"] = False
                    result["errors"].append(f"Unsupported audio format: {detected_content_type}")
            elif detected_content_type in SUPPORTED_DOCUMENT_TYPES:
                # Supported document type
                pass
            else:
                result["valid"] = False
                result["errors"].append(f"Unsupported content type: {detected_content_type}")
        else:
            # If we can't detect content type, check if it's a known file extension
            if file_ext in extension_map:
                result["warnings"].append(f"Could not detect MIME type, but file extension {file_ext} is supported")
            else:
                result["valid"] = False
                result["errors"].append(f"Unknown file type with extension: {file_ext}")
        
        return result
    except Exception as e:
        logger.error(f"Validation error: {e}")
        result["valid"] = False
        result["errors"].append(f"Validation error: {str(e)}")
        return result
        result["valid"] = False
        result["errors"].append(f"Validation error: {str(e)}")
        return result

def determine_processing_queue(content_type: str) -> str:
    """
    Determine which RabbitMQ queue to route a message to based on content type
    """
    content_type = content_type.lower()
    
    if content_type in ['image']:
        return 'image_processing_queue'
    elif content_type in ['video', 'audio']:
        return 'video_processing_queue'
    elif content_type in ['pdf', 'document']:
        return 'pdf_processing_queue'
    elif content_type in ['text', 'link']:
        return 'text_processing_queue'
    else:
        # Default to text processing for unknown types
        return 'text_processing_queue'

# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize database and RabbitMQ on startup"""
    global orchestrator
    try:
        init_database()
        logger.info("Database initialized successfully")
        
        # Initialize RabbitMQ orchestrator
        orchestrator = RabbitMQOrchestrator()
        logger.info("RabbitMQ Orchestrator initialized successfully")
        
        logger.info("Processing & Storage API with RabbitMQ initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global orchestrator
    if orchestrator:
        try:
            orchestrator.close()
            logger.info("RabbitMQ connections closed")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connections: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Content Processing & Storage API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/health") 
async def api_health():
    """API health check endpoint"""
    return {"status": "healthy", "api_version": "1.0.0", "timestamp": datetime.now().isoformat()}

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "Test endpoint working", "timestamp": datetime.now().isoformat()}

@app.post("/api/process-message", response_model=ProcessingResponse)
async def process_message(request: ProcessingRequest):
    """Process and store text message"""
    try:
        logger.info(f"=== INCOMING REQUEST ===")
        logger.info(f"Raw request: {request}")
        logger.info(f"Request dict: {request.dict()}")
        logger.info(f"User identifier: '{request.user_identifier}'")
        logger.info(f"Chat JID: '{request.chat_jid}'")
        logger.info(f"Sender JID: '{request.sender_jid}'")
        logger.info(f"Content: '{request.content}'")
        logger.info(f"========================")
        
        logger.info(f"Processing message from {request.user_identifier}: {request.content_type}")
          # Create a simple object with safe attribute access
        class SimpleRequest:
            def __init__(self, pydantic_request):
                # Convert Pydantic model to dict and then set attributes
                request_dict = pydantic_request.dict()
                for key, value in request_dict.items():
                    setattr(self, key, value)
        req = SimpleRequest(request)
        
        # Debug: Check what attributes the SimpleRequest has
        logger.info(f"SimpleRequest attributes: {vars(req)}")
        logger.info(f"chat_jid from SimpleRequest: {getattr(req, 'chat_jid', 'NOT_FOUND')}")        # Store in database
        stored_id = store_in_raw_database(req)
        
        if not stored_id:
            return ProcessingResponse(
                success=False,
                message="Failed to store in database",
                error="Database storage error",
                should_retry=True
            )
        
        # Route to RabbitMQ for processing
        try:
            if orchestrator:
                # Determine queue based on content type
                queue_name = determine_processing_queue(request.content_type)
                orchestrator.route_to_queue(stored_id, queue_name)
                logger.info(f"Message {stored_id} routed to {queue_name}")
                
                return ProcessingResponse(
                    success=True,
                    message="Message processed and queued for analysis",
                    stored_id=stored_id,                    processing_job_id=stored_id,
                    validation_result={"status": "queued"},
                    should_retry=False
                )
            else:
                logger.warning("RabbitMQ orchestrator not available, processing without queue")
                return ProcessingResponse(
                    success=True,
                    message="Message processed and stored successfully (no queue)",
                    stored_id=stored_id,
                    processing_job_id=generate_uuid(),
                    should_retry=False
                )
        except Exception as queue_error:
            logger.error(f"RabbitMQ routing failed: {queue_error}")
            return ProcessingResponse(
                success=False,
                message="Message stored but queue routing failed",
                stored_id=stored_id,
                error=f"RabbitMQ routing failed: {str(queue_error)}",
                should_retry=True
            )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ProcessingResponse(
            success=False,
            message="Processing failed",
            error=str(e),
            should_retry=True
        )

@app.post("/api/process-file", response_model=ProcessingResponse)
async def process_file(
    file: UploadFile = File(...),
    user_identifier: str = Form(...),
    message_id: str = Form(...),
    chat_jid: str = Form(...),
    chat_name: str = Form(...),
    sender_jid: str = Form(...),
    sender_name: str = Form(...),
    timestamp: str = Form(...),
    content: str = Form(default=""),
    content_type: str = Form(default="document"),
    is_from_me: str = Form(default="false"),
    is_group: str = Form(default="false"),
    source_type: str = Form(default="whatsapp"),
    priority: str = Form(default="normal")
):    
    
    stored_uuid = None
    temp_path = None
    
    try:
        logger.info(f"Processing file upload from {user_identifier}: {file.filename}")
        logger.info(f"🔍 Received content_type: {file.content_type}")
        logger.info(f"🔍 File size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # Handle boolean fields properly
        is_from_me_bool = is_from_me.lower() in ("true", "t", "1", "yes", "y")
        is_group_bool = is_group.lower() in ("true", "t", "1", "yes", "y")
        
        # Save file temporarily
        temp_path = await save_uploaded_file(file, user_identifier)
        
        # Validate file (ignore content_type from upload, detect from file extension)
        validation_result = validate_file(temp_path, None)
        if not validation_result["valid"]:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return ProcessingResponse(
                success=False,
                message="File validation failed",
                validation_result=validation_result,
                error=", ".join(validation_result["errors"]),
                should_retry=False
            )
        
        # Store file in permanent storage
        file_storage_path = store_file_in_storage(temp_path, user_identifier, file.filename)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Create request object with safe type conversion
        class SimpleRequest:
            def __init__(self):
                self.message_id = safe_convert_value(message_id, 'message_id')
                self.chat_jid = safe_convert_value(chat_jid, 'chat_jid')
                self.chat_name = safe_convert_value(chat_name, 'chat_name')
                self.sender_jid = sender_jid
                self.sender_name = safe_convert_value(sender_name, 'sender_name')
                self.user_identifier = safe_convert_value(user_identifier, 'user_identifier')
                self.content = safe_convert_value(content, 'content')
                self.content_type = safe_convert_value(content_type, 'content_type')
                self.media_filename = safe_convert_value(file.filename, 'media_filename')
                self.media_size = safe_convert_value(validation_result["file_size"], 'media_size')
                self.media_path = safe_convert_value(file_storage_path, 'media_path')
                self.is_from_me = safe_convert_value(is_from_me_bool, 'is_from_me')
                self.is_group = safe_convert_value(is_group_bool, 'is_group')
                self.timestamp = safe_convert_value(timestamp, 'timestamp')
                self.source_type = safe_convert_value(source_type, 'source_type')
                self.priority = safe_convert_value(priority, 'priority')
        
        req = SimpleRequest()        # Store in database
        stored_id = store_in_raw_database(req, file_storage_path)
        
        if not stored_id:
            # If storage failed, remove the file
            if file_storage_path and os.path.exists(file_storage_path):
                os.remove(file_storage_path)
                logger.info(f"Removed file {file_storage_path} due to database storage failure")
                
            return ProcessingResponse(
                success=False,
                message="Failed to store in database",
                error="Database storage error",
                should_retry=True
            )
        
        # Route file to RabbitMQ for processing after successful storage
        try:
            if orchestrator:
                # Determine the appropriate queue based on content type
                queue_name = determine_processing_queue(req.content_type)
                
                # Route to RabbitMQ using the correct method signature
                orchestrator.route_to_queue(stored_id, queue_name)
                logger.info(f"File message routed to {queue_name} for processing")
                message_status = f"File stored and queued for {queue_name.replace('_queue', '')} analysis"
            else:
                logger.warning("RabbitMQ orchestrator not available")
                message_status = "File stored but analysis service unavailable"
        except Exception as e:
            logger.error(f"Error routing file to RabbitMQ: {e}")
            message_status = "File stored but analysis routing failed"
        
        return ProcessingResponse(
            success=True,
            message=message_status,
            stored_id=stored_id,
            file_storage_path=file_storage_path,
            validation_result=validation_result,
            processing_job_id=generate_uuid(),
            should_retry=False
        )
        
    except Exception as e:
        logger.error(f"File processing error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Clean up resources on error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            
        return ProcessingResponse(
            success=False,
            message="File processing failed",
            error=str(e),
            should_retry=True
        )

# Add additional endpoints for database management
@app.get("/api/records")
async def get_records(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None
):
    """Get records from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        if status:
            cursor.execute('''
                SELECT * FROM raw_data WHERE processing_status = ?
                ORDER BY submission_timestamp DESC LIMIT ? OFFSET ?
            ''', (status, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM raw_data ORDER BY submission_timestamp DESC LIMIT ? OFFSET ?
            ''', (limit, offset))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return {"data": results, "count": len(results)}
        
    finally:
        conn.close()

@app.get("/api/pending")
async def get_pending_record():
    """Get the next pending record for extraction"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT * FROM raw_data WHERE processing_status = 'pending'
            ORDER BY submission_timestamp ASC LIMIT 1
        ''')
        
        row = cursor.fetchone()
        if row:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))
        else:
            return {"message": "No pending records found"}
            
    finally:
        conn.close()

@app.put("/api/record/{uuid}/status")
async def update_record_status(uuid: str, status: str):
    """Update status of a record"""
    valid_statuses = ["pending", "processing", "completed", "failed"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE raw_data SET processing_status = ? WHERE UUID = ?
        ''', (status, uuid))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        
        conn.commit()
        return {"message": f"Status updated to {status}", "uuid": uuid}
        
    finally:
        conn.close()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        
        # Check directories
        directories = {
            "uploads": os.path.exists(UPLOAD_DIRECTORY),
            "processed": os.path.exists(PROCESSED_DIRECTORY),
            "file_storage": os.path.exists(FILE_STORAGE_PATH),
            "databases": os.path.exists("databases")
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": {
                "accessible": True,
                "path": DB_PATH
            },
            "directories": directories,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Run the application
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Content Processing & Storage API on http://0.0.0.0:8001")
    print("📋 Available endpoints:")
    print("   - GET  /health")
    print("   - GET  /api/health") 
    print("   - POST /api/process-message")
    print("   - POST /api/process-file")
    print("🔥 API is ready to receive requests!")
    uvicorn.run(app, host="0.0.0.0", port=8001)