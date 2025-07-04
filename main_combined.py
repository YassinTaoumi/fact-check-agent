# main_combined.py - FastAPI application with direct RabbitMQ pipeline integration
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import os
import logging
import httpx
import asyncio
import json
import sqlite3
import uuid
import shutil
from pathlib import Path

# Import RabbitMQ orchestrator components
from rabbitmq_orchestrator import RabbitMQOrchestrator, MessageRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhatsApp Message Processing API", 
    description="Direct RabbitMQ pipeline integration for message processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for RabbitMQ pipeline
RABBITMQ_ORCHESTRATOR_URL = os.getenv("RABBITMQ_ORCHESTRATOR_URL", "http://localhost:8002")
DB_PATH = os.getenv("DB_PATH", "databases/raw_data.db")
FILE_STORAGE_DIR = os.getenv("FILE_STORAGE_DIR", "file_storage")

# Timeout for API calls
API_TIMEOUT = 30.0

# Initialize RabbitMQ orchestrator
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize RabbitMQ orchestrator and setup database on startup"""
    global orchestrator
    try:
        # Setup database first
        setup_database()
        logger.info("Database setup completed")
        
        # Initialize RabbitMQ orchestrator
        orchestrator = RabbitMQOrchestrator()
        logger.info("RabbitMQ Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        # Don't fail startup, but log the error
        orchestrator = None

def setup_database():
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
                        metadata TEXT DEFAULT '{}'
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

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean up on shutdown"""
    global orchestrator
    if orchestrator:
        orchestrator.close_connection()
    logger.info("Application shutdown completed")

# Pydantic models for API
class APIProcessingRequest(BaseModel):
    message_id: str
    chat_jid: str
    chat_name: str
    sender_jid: str
    sender_name: str
    user_identifier: str
    content: str
    content_type: str
    media_filename: Optional[str] = None
    media_size: Optional[int] = None
    media_path: Optional[str] = None
    is_from_me: bool = False
    is_group: bool = False
    timestamp: str
    source_type: Optional[str] = "whatsapp"
    priority: Optional[str] = "normal"

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    stored_id: Optional[str] = None
    file_storage_path: Optional[str] = None
    processing_job_id: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    should_retry: bool = False

class FileProcessingRequest(BaseModel):
    user_identifier: str
    message_id: str
    chat_jid: str
    chat_name: str
    sender_jid: str
    sender_name: str
    timestamp: str
    content: Optional[str] = ""
    content_type: Optional[str] = "document"
    is_from_me: Optional[bool] = False
    is_group: Optional[bool] = False
    source_type: Optional[str] = "whatsapp"
    priority: Optional[str] = "normal"

class HealthStatus(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Dict[str, Any]]
    overall_health: bool

# Helper Functions
def ensure_file_storage_dir():
    """Ensure file storage directory exists"""
    os.makedirs(FILE_STORAGE_DIR, exist_ok=True)

def save_uploaded_file(file_content: bytes, filename: str, user_identifier: str) -> str:
    """Save uploaded file to storage and return path"""
    ensure_file_storage_dir()
    
    # Create user-specific directory
    user_dir = os.path.join(FILE_STORAGE_DIR, user_identifier)
    os.makedirs(user_dir, exist_ok=True)
    
    # Generate unique filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(filename).suffix
    unique_filename = f"{timestamp}_{filename}"
    
    file_path = os.path.join(user_dir, unique_filename)
    
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    return file_path

def store_message_in_database(message_record: MessageRecord) -> str:
    """Store message directly in database and return the generated ID"""
    try:
        # Generate unique identifiers
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        message_uuid = str(uuid.uuid4())
        
        # Prepare metadata as JSON string
        metadata_json = json.dumps(message_record.metadata) if message_record.metadata else None
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO raw_data (
                    ID, UUID, source_type, sender_phone, is_group_message,
                    group_name, channel_name, chat_jid, content_type, content_url,
                    raw_text, submission_timestamp, processing_status,
                    user_identifier, priority, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id, message_uuid, message_record.source_type, message_record.sender_phone,
                message_record.is_group_message, message_record.group_name, message_record.channel_name,
                message_record.chat_jid, message_record.content_type, message_record.content_url,
                message_record.raw_text, datetime.now().isoformat(), 'pending',
                message_record.user_identifier, message_record.priority, metadata_json
            ))
            conn.commit()
        
        logger.info(f"Message stored in database with ID: {message_id}")
        return message_id
        
    except Exception as e:
        logger.error(f"Failed to store message in database: {e}")
        raise HTTPException(status_code=500, detail=f"Database storage failed: {str(e)}")

def get_stored_record(message_id: str) -> dict:
    """Retrieve the stored record from database by ID"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM raw_data WHERE ID = ?", (message_id,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Message with ID {message_id} not found in database")
            
            # Convert row to dictionary
            record = dict(row)
            
            # Parse metadata back from JSON
            if record['metadata']:
                record['metadata'] = json.loads(record['metadata'])
            
            return record
            
    except Exception as e:
        logger.error(f"Failed to retrieve stored record: {e}")
        raise HTTPException(status_code=500, detail=f"Database retrieval failed: {str(e)}")

async def send_to_rabbitmq_for_analysis(stored_record: dict) -> dict:
    """Send stored record to RabbitMQ pipeline for analysis"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=500, detail="RabbitMQ orchestrator not initialized")
    
    try:
        # Determine appropriate queue based on content type
        queue_name = orchestrator.get_processing_queue(stored_record['content_type'])
        
        # Route the stored record to the processing queue
        orchestrator.route_to_queue(stored_record['ID'], queue_name)
        
        logger.info(f"Record {stored_record['ID']} sent to RabbitMQ queue: {queue_name}")
        
        return {
            "success": True,
            "message_id": stored_record['ID'],
            "queue": queue_name,
            "status": "queued_for_analysis"
        }
        
    except Exception as e:
        logger.error(f"Failed to send record to RabbitMQ: {e}")
        raise HTTPException(status_code=500, detail=f"RabbitMQ routing failed: {str(e)}")

async def call_rabbitmq_orchestrator(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make API call to RabbitMQ Orchestrator API for health checks and status"""
    url = f"{RABBITMQ_ORCHESTRATOR_URL}{endpoint}"
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            if method == "POST":
                response = await client.post(url, json=data)
            elif method == "GET":
                response = await client.get(url, params=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
    except httpx.TimeoutException:
        logger.error(f"Timeout calling RabbitMQ Orchestrator: {url}")
        raise HTTPException(status_code=504, detail="RabbitMQ Orchestrator timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling RabbitMQ Orchestrator: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"RabbitMQ Orchestrator error: {e.response.text}")
    except Exception as e:
        logger.error(f"Error calling RabbitMQ Orchestrator: {e}")
        raise HTTPException(status_code=500, detail=f"RabbitMQ Orchestrator error: {str(e)}")

def determine_message_context(chat_jid: str, chat_name: str, is_group: bool) -> Dict[str, Any]:
    """Determine message context and extract group/channel information"""
    context = {
        "is_group_message": is_group,
        "is_group": False,
        "group_name": None,
        "is_channel": False,
        "channel_name": None
    }
    
    # if is_group and chat_jid and chat_name:
    #     # Check if it's a WhatsApp channel
    #     if '@newsletter' in chat_jid:
    #         context["is_channel"] = True
    #         context["channel_name"] = chat_name
    #         logger.info(f"Detected WhatsApp channel: {chat_name}")
    #     elif '@g.us' in chat_jid:
    #         # Regular WhatsApp group
    #         context["is_group"] = True
    #         context["group_name"] = chat_name
    #         logger.info(f"Detected WhatsApp group: {chat_name}")
    
    return context

# Main API Endpoints
@app.post("/api/process-message", response_model=ProcessingResponse)
async def process_message(req: APIProcessingRequest):
    """Main endpoint for processing incoming messages - direct RabbitMQ pipeline integration"""
    
    try:
        logger.info(f"Processing message from {req.user_identifier}: {req.content_type}")
        logger.info(f"DEBUG: Request data: {req}")
        
        message_context = determine_message_context(req.chat_jid, req.chat_name, req.is_group)
        
        # Create MessageRecord for RabbitMQ orchestrator
        message_record = MessageRecord(
            source_type=req.source_type or "whatsapp",
            sender_phone=req.sender_jid,  # Using sender_jid as phone number
            is_group_message=req.is_group,
            group_name=req.chat_name if req.is_group else None,
            channel_name=None,  # Could be extracted if needed
            chat_jid=req.chat_jid,
            content_type=req.content_type,
            content_url=req.media_path,
            raw_text=req.content,
            user_identifier=req.user_identifier,
            priority=req.priority or "normal",            metadata={
                "message_id": req.message_id,
                "sender_name": req.sender_name,
                "chat_name": req.chat_name,
                "timestamp": req.timestamp,
                "is_from_me": req.is_from_me,
                "media_filename": req.media_filename,
                "media_size": req.media_size
            }
        )
        
        # Step 1: Store message in database first
        message_id = store_message_in_database(message_record)
        
        # Step 2: Retrieve the stored record with its ID
        stored_record = get_stored_record(message_id)
        
        # Step 3: Send stored record to RabbitMQ for analysis
        result = await send_to_rabbitmq_for_analysis(stored_record)
        
        if not result.get("success", False):
            return ProcessingResponse(
                success=False,
                message="RabbitMQ processing failed",
                error=result.get("error", "Unknown error"),
                should_retry=True
            )
        
        # Create success message with context
        success_message = "Message processed and queued for analysis"
        if message_context.get("is_group"):
            success_message += f" from group: {message_context.get('group_name')}"
        elif message_context.get("is_channel"):
            success_message += f" from channel: {message_context.get('channel_name')}"
        
        return ProcessingResponse(
            success=True,
            message=success_message,
            stored_id=result.get("message_id"),
            file_storage_path=req.media_path,
            processing_job_id=result.get("message_id"),
            validation_result={"status": "queued"},
            should_retry=False
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Message processing error: {e}")
        return ProcessingResponse(
            success=False,
            message="Processing failed",
            error=str(e),
            should_retry=True
        )


@app.post("/api/process-file", response_model=ProcessingResponse)
async def process_file_upload(
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
    media_filename: str = Form(default=""),
    media_size: int = Form(default=0),
    is_from_me: bool = Form(default=False),
    is_group: bool = Form(default=False),
    source_type: str = Form(default="whatsapp"),
    priority: str = Form(default="normal")
):
    """Process file upload using the combined API"""
    
    try:
        logger.info(f"Processing file upload from {user_identifier}: {file.filename}")
        logger.info("FileProcessingRequest Variable Types:")
        logger.info(f"  user_identifier: {type(user_identifier)} ({user_identifier})")
        logger.info(f"  message_id:      {type(message_id)} ({message_id})")
        logger.info(f"  chat_jid:        {type(chat_jid)} ({chat_jid})")
        logger.info(f"  chat_name:       {type(chat_name)} ({chat_name})")
        logger.info(f"  sender_jid:      {type(sender_jid)} ({sender_jid})")
        logger.info(f"  sender_name:     {type(sender_name)} ({sender_name})")
        logger.info(f"  timestamp:       {type(timestamp)} ({timestamp})")
        logger.info(f"  content:         {type(content)} ({content[:30] if content else ''})")
        logger.info(f"  content_type:    {type(content_type)} ({content_type})")
        logger.info(f"  media_filename:  {type(media_filename)} ({media_filename})")
        logger.info(f"  media_size:      {type(media_size)} ({media_size})")
        logger.info(f"  is_from_me:      {type(is_from_me)} ({is_from_me})")
        logger.info(f"  is_group:        {type(is_group)} ({is_group})")
        logger.info(f"  source_type:     {type(source_type)} ({source_type})")
        logger.info(f"  priority:        {type(priority)} ({priority})")
          # Save uploaded file
        file_content = await file.read()
        file_path = save_uploaded_file(file_content, file.filename, user_identifier)
        logger.info(f"File saved to: {file_path}")
        
        # Create MessageRecord for RabbitMQ orchestrator
        message_record = MessageRecord(
            source_type=source_type,
            sender_phone=sender_jid,
            is_group_message=is_group,
            group_name=chat_name if is_group else None,
            channel_name=None,
            chat_jid=chat_jid,
            content_type=content_type,
            content_url=file_path,
            raw_text=content,
            user_identifier=user_identifier,
            priority=priority,            metadata={
                "message_id": message_id,
                "sender_name": sender_name,
                "chat_name": chat_name,
                "timestamp": timestamp,
                "is_from_me": is_from_me,
                "media_filename": file.filename,
                "media_size": len(file_content),
                "original_filename": file.filename,
                "content_type_header": file.content_type
            }
        )
        
        # Step 1: Store message in database first
        stored_message_id = store_message_in_database(message_record)
        
        # Step 2: Retrieve the stored record with its ID
        stored_record = get_stored_record(stored_message_id)
        
        # Step 3: Send stored record to RabbitMQ for analysis
        result = await send_to_rabbitmq_for_analysis(stored_record)
        
        if not result.get("success", False):
            return ProcessingResponse(
                success=False,
                message="File processing and queueing failed",
                error=result.get("error", "Unknown processing error"),
                should_retry=True
            )
        
        return ProcessingResponse(
            success=True,
            message="File processed and queued for analysis",
            stored_id=result.get("message_id"),
            file_storage_path=file_path,
            processing_job_id=result.get("message_id"),
            validation_result={"status": "queued", "file_saved": True},
            should_retry=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File processing error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ProcessingResponse(
            success=False,
            message="File processing failed",
            error=str(e),
            should_retry=True
        )

# Database access endpoints
@app.get("/api/records")
async def get_records(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None
):
    """Get records from the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM raw_data"
            params = []
            
            if status:
                query += " WHERE processing_status = ?"
                params.append(status)
            
            query += " ORDER BY submission_timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            records = [dict(row) for row in rows]
            
            return {
                "records": records,
                "total": len(records),
                "limit": limit,
                "offset": offset
            }
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/pending")
async def get_pending_record():
    """Get the next pending record for extraction"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT * FROM raw_data 
                WHERE processing_status = 'pending' 
                ORDER BY submission_timestamp ASC 
                LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row:
                return {"record": dict(row)}
            else:
                return {"record": None, "message": "No pending records"}
                
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.put("/api/record/{record_id}/status")
async def update_record_status(record_id: str, status: str):
    """Update status of a record"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("""
                UPDATE raw_data 
                SET processing_status = ? 
                WHERE ID = ?
            """, (status, record_id))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Record not found")
            
            conn.commit()
            
            return {"success": True, "message": f"Status updated to {status}"}
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/record/{record_id}")
async def get_record(record_id: str):
    """Get a specific record by ID"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("SELECT * FROM raw_data WHERE ID = ?", (record_id,))
            row = cursor.fetchone()
            
            if row:
                return {"record": dict(row)}
            else:
                raise HTTPException(status_code=404, detail="Record not found")
                
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/api/record/{record_id}")
async def delete_record(record_id: str, permanent: bool = False):
    """Delete a record"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("DELETE FROM raw_data WHERE ID = ?", (record_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Record not found")
            
            conn.commit()
            
            return {"success": True, "message": "Record deleted"}
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/files/{user_identifier}")
async def get_user_files(user_identifier: str):
    """Get files for a specific user"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT * FROM raw_data 
                WHERE user_identifier = ? AND content_url IS NOT NULL
                ORDER BY submission_timestamp DESC
            """, (user_identifier,))
            rows = cursor.fetchall()
            
            files = [dict(row) for row in rows]
            
            return {"files": files, "user": user_identifier}
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/supported-formats")
async def get_supported_formats():
    """Get supported file formats"""
    return {
        "image_formats": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"],
        "video_formats": [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm"],
        "audio_formats": [".mp3", ".wav", ".ogg", ".aac", ".flac"],
        "document_formats": [".pdf", ".doc", ".docx", ".txt", ".rtf"],
        "archive_formats": [".zip", ".rar", ".7z", ".tar", ".gz"]
    }

@app.get("/api/validate-file/{file_path:path}")
async def validate_file(file_path: str):
    """Validate a file without processing it"""
    try:
        if not os.path.exists(file_path):
            return {"valid": False, "error": "File not found"}
        
        file_size = os.path.getsize(file_path)
        file_extension = Path(file_path).suffix.lower()
        
        # Basic validation
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            return {"valid": False, "error": "File too large"}
        
        supported_formats = await get_supported_formats()
        all_formats = []
        for format_list in supported_formats.values():
            all_formats.extend(format_list)
        
        if file_extension not in all_formats:
            return {"valid": False, "error": "Unsupported file format"}
        
        return {
            "valid": True,
            "file_size": file_size,
            "file_extension": file_extension,
            "file_path": file_path
        }
        
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return {"valid": False, "error": str(e)}

# Group and channel endpoints
@app.get("/api/groups")
async def get_groups_endpoint():
    """Get all groups that have sent messages"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT DISTINCT group_name, chat_jid, COUNT(*) as message_count,
                       MAX(submission_timestamp) as last_message
                FROM raw_data 
                WHERE is_group_message = 1 AND group_name IS NOT NULL
                GROUP BY group_name, chat_jid
                ORDER BY last_message DESC
            """)
            rows = cursor.fetchall()
            
            groups = [dict(row) for row in rows]
            
            return {
                "groups": groups,
                "total": len(groups)
            }
            
    except Exception as e:
        logger.error(f"Error getting groups: {e}")
        return {
            "message": "Error retrieving groups",
            "groups": [],
            "error": str(e)
        }

@app.get("/api/channels")
async def get_channels_endpoint():
    """Get all channels that have sent messages"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT DISTINCT channel_name, chat_jid, COUNT(*) as message_count,
                       MAX(submission_timestamp) as last_message
                FROM raw_data 
                WHERE channel_name IS NOT NULL
                GROUP BY channel_name, chat_jid
                ORDER BY last_message DESC
            """)
            rows = cursor.fetchall()
            
            channels = [dict(row) for row in rows]
            
            return {
                "channels": channels,
                "total": len(channels)
            }
            
    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        return {
            "message": "Error retrieving channels",
            "channels": [],
            "error": str(e)
        }

# Health and Stats endpoints
@app.get("/api/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check for all services"""
    
    services = {}
    overall_health = True
    
    # Check RabbitMQ Orchestrator
    try:
        rabbitmq_health = await call_rabbitmq_orchestrator("/health", "GET")
        services["rabbitmq_orchestrator"] = {
            "status": "healthy",
            "url": RABBITMQ_ORCHESTRATOR_URL,
            "response": rabbitmq_health
        }
    except Exception as e:
        services["rabbitmq_orchestrator"] = {
            "status": "unhealthy",
            "url": RABBITMQ_ORCHESTRATOR_URL,
            "error": str(e)
        }
        overall_health = False
    
    # Check Database
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM raw_data")
            count = cursor.fetchone()[0]
            services["database"] = {
                "status": "healthy",
                "path": DB_PATH,
                "total_records": count
            }
    except Exception as e:
        services["database"] = {
            "status": "unhealthy",
            "path": DB_PATH,
            "error": str(e)
        }
        overall_health = False
    
    # Check File Storage
    try:
        if os.path.exists(FILE_STORAGE_DIR):
            services["file_storage"] = {
                "status": "healthy",
                "path": FILE_STORAGE_DIR,
                "exists": True
            }
        else:
            services["file_storage"] = {
                "status": "warning",
                "path": FILE_STORAGE_DIR,
                "exists": False,
                "message": "Directory will be created when needed"
            }
    except Exception as e:
        services["file_storage"] = {
            "status": "unhealthy",
            "path": FILE_STORAGE_DIR,
            "error": str(e)
        }
    
    return HealthStatus(
        status="healthy" if overall_health else "degraded",
        timestamp=datetime.now().isoformat(),
        services=services,
        overall_health=overall_health
    )

@app.get("/api/stats")
async def get_stats_endpoint():
    """Get comprehensive statistics"""
    
    try:
        stats = {}
        
        # Database stats
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            # Total records
            cursor = conn.execute("SELECT COUNT(*) as total FROM raw_data")
            stats["total_records"] = cursor.fetchone()[0]
            
            # Records by status
            cursor = conn.execute("""
                SELECT processing_status, COUNT(*) as count 
                FROM raw_data 
                GROUP BY processing_status
            """)
            stats["by_status"] = {row["processing_status"]: row["count"] for row in cursor.fetchall()}
            
            # Records by content type
            cursor = conn.execute("""
                SELECT content_type, COUNT(*) as count 
                FROM raw_data 
                GROUP BY content_type
            """)
            stats["by_content_type"] = {row["content_type"]: row["count"] for row in cursor.fetchall()}
            
            # Records by source
            cursor = conn.execute("""
                SELECT source_type, COUNT(*) as count 
                FROM raw_data 
                GROUP BY source_type
            """)
            stats["by_source"] = {row["source_type"]: row["count"] for row in cursor.fetchall()}
        
        # RabbitMQ queue stats (if available)
        try:
            queue_stats = await call_rabbitmq_orchestrator("/queues/status", "GET")
            stats["queues"] = queue_stats.get("queues", {})
        except Exception as e:
            stats["queues"] = {"error": str(e)}
        
        return {
            "stats": stats,
            "features_enabled": {
                "group_detection": True,
                "channel_detection": True,
                "link_detection": True,
                "file_validation": True,
                "rabbitmq_processing": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}

@app.get("/api/system-status")
async def get_system_status():
    """Get overall system status and configuration"""
    return {
        "service": "whatsapp-message-processing-api",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "rabbitmq_orchestrator_url": RABBITMQ_ORCHESTRATOR_URL,
            "database_path": DB_PATH,
            "file_storage_dir": FILE_STORAGE_DIR,
            "api_timeout": API_TIMEOUT
        },
        "endpoints": {
            "rabbitmq_orchestrator": f"{RABBITMQ_ORCHESTRATOR_URL}/docs",
            "main_api": "/docs"
        },
        "features": {
            "group_detection": True,
            "channel_detection": True,
            "link_detection": True,
            "file_processing": True,
            "content_validation": True,
            "rabbitmq_processing": True,
            "direct_database_access": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
