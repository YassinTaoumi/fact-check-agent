# storage_api.py - Complete Storage Management API with integrated functions
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, BackgroundTasks,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from enum import Enum
import os
import json
import sqlite3
import shutil
import hashlib
import uuid
import logging
from pathlib import Path
import aiofiles
import uvicorn
from types import SimpleNamespace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database and storage paths
DB_PATH = "databases/raw_data.db"
FILE_STORAGE_PATH = "file_storage"

# Function to check and print database schema
def debug_database_schema():
    try:
        if not os.path.exists(DB_PATH):
            logger.warning(f"DEBUG: Database file does not exist: {DB_PATH}")
            return
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(raw_data)")
        columns = cursor.fetchall()
        logger.info(f"DEBUG: Current database schema for raw_data: {columns}")
        conn.close()
    except Exception as e:
        logger.error(f"DEBUG: Error checking schema: {e}")

# Print database schema on startup
debug_database_schema()

# ================================
# STORAGE FUNCTIONS (Must be defined FIRST)
# ================================

def init_database():
    """Initialize raw data database"""
    os.makedirs("databases", exist_ok=True)
    os.makedirs(FILE_STORAGE_PATH, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS raw_data (
        ID TEXT PRIMARY KEY,
        UUID TEXT NOT NULL,
        source_type TEXT NOT NULL CHECK (source_type IN ('whatsapp', 'telegram')),
        sender_phone TEXT NOT NULL,
        is_group_message BOOLEAN NOT NULL DEFAULT 0,
        group_name TEXT,
        channel_name TEXT,
        content_type TEXT NOT NULL CHECK (content_type IN ('audio', 'video', 'pdf', 'image', 'text', 'document', 'link')),
        content_url TEXT,
        raw_text TEXT,
        submission_timestamp DATETIME NOT NULL,
        processing_status TEXT NOT NULL DEFAULT 'pending',
        user_identifier TEXT NOT NULL,
        priority TEXT NOT NULL DEFAULT 'normal'
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

def store_in_raw_database(req, file_storage_path: str = None) -> str:
    """Store message data in raw database with simplified fields"""
    
    record_id = generate_id(req.message_id, req.chat_jid, req.timestamp)
    record_uuid = generate_uuid()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Extract sender phone number from sender_jid        sender_phone = getattr(req, 'user_identifier', '')
        if not sender_phone and hasattr(req, 'sender_jid'):
            # Extract phone number from JID (e.g., "123456789@s.whatsapp.net" -> "123456789")
            sender_phone = req.sender_jid.split('@')[0]
        
        # Determine if it's a group message and extract group/channel name
        is_group_message = getattr(req, 'is_group', False)
        group_name = None
        channel_name = None
        
        if is_group_message:
            # For WhatsApp groups, chat_name contains the group name
            group_name = getattr(req, 'chat_name', '')
            # Check if it's actually a channel (WhatsApp channels have different JID patterns)
            chat_jid = getattr(req, 'chat_jid', '')
            if '@newsletter' in chat_jid:
                # This is a WhatsApp channel, not a group
                channel_name = group_name
                group_name = None
                is_group_message = False
        
        logger.info(f"DEBUG: Request object fields: {dir(req)}")
        logger.info(f"DEBUG: Request object dict: {req.__dict__ if hasattr(req, '__dict__') else 'No __dict__'}")
        logger.info(f"Storing record: sender_phone={sender_phone}, is_group_message={is_group_message}, group_name={group_name}, channel_name={channel_name}")
        if not sender_phone:
            logger.error("sender_phone is empty, skipping insert to avoid NOT NULL constraint error.")
            return None
        
        # Debug the SQL statement before executing
        sql_statement = '''
            INSERT OR REPLACE INTO raw_data (
                ID, UUID, source_type, sender_phone, is_group_message, group_name, 
                channel_name, content_type, content_url, raw_text, 
                submission_timestamp, processing_status, user_identifier, priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''        
        values = (
            record_id,
            record_uuid,
            getattr(req, 'source_type', 'whatsapp'),
            sender_phone,
            is_group_message,
            group_name,
            channel_name,
            req.content_type,
            file_storage_path,
            getattr(req, 'content', ''),
            req.timestamp,
            "pending",
            req.user_identifier,
            getattr(req, 'priority', 'normal')
        )
        
        logger.info(f"DEBUG: SQL: {sql_statement}")
        logger.info(f"DEBUG: Values: {values}")
          # Check if req has source_metadata
        if hasattr(req, 'source_metadata'):            
            logger.error("CRITICAL: req object has source_metadata attribute!")
            logger.error(f"source_metadata value: {getattr(req, 'source_metadata', None)}")
            # Remove it to prevent issues
            delattr(req, 'source_metadata')
            
        try:
            safe_execute(cursor, sql_statement, values)
            conn.commit()
            logger.info(f"Stored raw data with ID: {record_id}, UUID: {record_uuid}")
            if file_storage_path:
                logger.info(f"Content URL set to: {file_storage_path}")
            return record_uuid
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            # Print the schema of the raw_data table to debug
            logger.info("DEBUG: Checking database schema...")
            cursor.execute("PRAGMA table_info(raw_data)")
            columns = cursor.fetchall()
            logger.info(f"DEBUG: Database columns: {columns}")
            raise
            
    except sqlite3.Error as e:
        logger.error(f"Database error storing raw data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error storing raw data: {e}")
        raise
    finally:
        conn.close()


def get_raw_data(limit: int = 20, offset: int = 0, status: str = None):
    """Get raw data records"""
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

def get_latest_pending():
    """Get the latest pending record for extraction processing"""
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

def update_processing_status(uuid: str, status: str):
    """Update processing status of a record"""
    valid_statuses = ["pending", "processing", "completed", "failed"]
    if status not in valid_statuses:
        raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE raw_data SET processing_status = ? WHERE UUID = ?
        ''', (status, uuid))
        
        if cursor.rowcount == 0:
            raise ValueError("Record not found")
        
        conn.commit()
        return {"message": f"Status updated to {status}", "uuid": uuid}
        
    finally:
        conn.close()

def get_stats():
    """Get processing statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Count by status
        cursor.execute('''
            SELECT processing_status, COUNT(*) as count 
            FROM raw_data 
            GROUP BY processing_status
        ''')
        status_counts = dict(cursor.fetchall())
        
        # Count by content type
        cursor.execute('''
            SELECT content_type, COUNT(*) as count 
            FROM raw_data 
            GROUP BY content_type
        ''')
        content_type_counts = dict(cursor.fetchall())
        
        # Count by message type
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN is_group = 1 THEN 'group'
                    WHEN is_channel = 1 THEN 'channel'
                    ELSE 'private'
                END as message_type,
                COUNT(*) as count 
            FROM raw_data 
            GROUP BY message_type
        ''')
        message_type_counts = dict(cursor.fetchall())
        
        # Total records
        cursor.execute('SELECT COUNT(*) FROM raw_data')
        total_records = cursor.fetchone()[0]
        
        return {
            "total_records": total_records,
            "by_status": status_counts,
            "by_content_type": content_type_counts,
            "by_message_type": message_type_counts
        }
        
    finally:
        conn.close()

def get_storage_info() -> Dict[str, Any]:
    """Get storage directory information"""
    try:
        storage_info = {
            "database_exists": os.path.exists(DB_PATH),
            "database_size_bytes": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0,
            "file_storage_exists": os.path.exists(FILE_STORAGE_PATH),
            "file_storage_size_bytes": 0,
            "file_count": 0
        }
        
        if os.path.exists(FILE_STORAGE_PATH):
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(FILE_STORAGE_PATH):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
                        file_count += 1
            
            storage_info["file_storage_size_bytes"] = total_size
            storage_info["file_count"] = file_count
        
        return storage_info
    except Exception as e:
        logger.error(f"Error getting storage info: {e}")
        return {"error": str(e)}

async def save_uploaded_file(file: UploadFile, user_identifier: str) -> str:
    """Save uploaded file to temporary location"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(file.filename).suffix if file.filename else ""
    temp_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        async with aiofiles.open(temp_path, 'wb') as temp_file:
            content = await file.read()
            await temp_file.write(content)
        
        return temp_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def execute_safe_query(cursor, sql, params=None):
    """Execute SQL query with safety checks for source_metadata"""
    # Check if the SQL contains source_metadata
    if "source_metadata" in sql.lower():
        logger.error(f"BLOCKED SQL QUERY - contains source_metadata: {sql}")
        raise ValueError("SQL query contains source_metadata which is not in the schema")
    
    # Check parameter types to avoid the "Invalid type for value" error
    if params:
        for i, param in enumerate(params):
            if isinstance(param, dict):
                logger.error(f"PARAM TYPE ERROR: Parameter {i} is a dict: {param}")
                # Convert dict to string to avoid the error
                params = list(params)  # Convert tuple to list for modification
                params[i] = json.dumps(param)  # Convert dict to JSON string
                params = tuple(params)  # Convert back to tuple
            elif param is None:
                # Replace None with empty string for text columns
                logger.warning(f"PARAM WARNING: Parameter {i} is None, replacing with empty string")
                params = list(params)
                params[i] = ""
                params = tuple(params)
    
    # Execute the query
    if params:
        return cursor.execute(sql, params)
    else:
        return cursor.execute(sql)

def safe_execute(cursor, sql, params=None):
    """Execute SQL safely, checking for issues with source_metadata"""
    if sql is None:
        logger.error("NULL SQL query attempted")
        return None
    
    # Check if SQL contains source_metadata
    if "source_metadata" in sql.lower():
        logger.error(f"BLOCKED SQL QUERY - contains source_metadata: {sql}")
        # Remove source_metadata from SQL if it's a column in INSERT
        if "INSERT" in sql.upper():
            logger.info("Attempting to fix INSERT query with source_metadata")
            # Extract column names
            cols_start = sql.find("(") + 1
            cols_end = sql.find(")", cols_start)
            if cols_start > 0 and cols_end > cols_start:
                columns_str = sql[cols_start:cols_end]
                columns = [c.strip() for c in columns_str.split(",")]
                
                # Find source_metadata index
                try:
                    sm_index = -1
                    for i, col in enumerate(columns):
                        if "source_metadata" in col.lower():
                            sm_index = i
                            break
                    
                    if sm_index >= 0:
                        # Remove source_metadata from columns
                        columns.pop(sm_index)
                        
                        # Rebuild the SQL
                        new_cols = ", ".join(columns)
                        
                        # Adjust the values part
                        values_part = sql[cols_end+1:]
                        values_start = values_part.find("(") + 1
                        values_end = values_part.find(")", values_start)
                        if values_start > 0 and values_end > values_start:
                            values_str = values_part[values_start:values_end]
                            values = [v.strip() for v in values_str.split(",")]
                            
                            if sm_index < len(values):
                                values.pop(sm_index)
                            
                            new_values = ", ".join(values)
                            
                            # Reconstruct SQL
                            new_sql = f"{sql[:cols_start]}{new_cols}{sql[cols_end:cols_end+1 + values_start]}{new_values})"
                            logger.info(f"Fixed SQL: {new_sql}")
                            sql = new_sql
                            
                            # Adjust params if needed
                            if params and isinstance(params, (list, tuple)) and sm_index < len(params):
                                params_list = list(params)
                                params_list.pop(sm_index)
                                params = tuple(params_list)
                except Exception as e:
                    logger.error(f"Failed to fix SQL with source_metadata: {e}")
    
    # Check parameter types to avoid "Invalid type for value" errors
    if params:
        try:
            # Convert params to list to allow modifications
            params_list = list(params)
            modified = False
            
            for i, param in enumerate(params_list):
                if isinstance(param, dict):
                    logger.error(f"PARAM TYPE ERROR: Parameter {i} is a dict: {param}")
                    # Convert dict to string
                    params_list[i] = json.dumps(param)
                    modified = True
                elif param is None:
                    # Replace None with empty string for text columns
                    logger.warning(f"PARAM WARNING: Parameter {i} is None, replacing with empty string")
                    params_list[i] = ""
                    modified = True
            
            if modified:
                params = tuple(params_list)
        except Exception as e:
            logger.error(f"Error checking parameter types: {e}")
    
    # Execute the query with proper error handling
    try:
        if params:
            return cursor.execute(sql, params)
        else:
            return cursor.execute(sql)
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        logger.error(f"Failed SQL: {sql}")
        if params:
            logger.error(f"Params: {params}")
        
        # Special handling for "no such column" errors
        if "no such column" in str(e).lower() and "source_metadata" in str(e).lower():
            logger.error("Detected 'no such column: source_metadata' error. This indicates a schema mismatch.")
            logger.info("Checking current database schema...")
            try:
                schema_cursor = cursor.connection.cursor()
                schema_cursor.execute("PRAGMA table_info(raw_data)")
                columns = schema_cursor.fetchall()
                logger.info(f"Current schema: {columns}")
            except Exception as schema_err:
                logger.error(f"Failed to check schema: {schema_err}")
        
        raise

# ================================
# FASTAPI APPLICATION (Defined AFTER functions)
# ================================

app = FastAPI(
    title="Storage Management API",
    description="API for managing file storage and database operations",
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

# Enums
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SourceType(str, Enum):
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"

class ContentType(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    DOCUMENT = "document"
    LINK = "link"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# Pydantic Models
class StorageRequest(BaseModel):
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
    priority: Priority = Priority.NORMAL
    source_type: SourceType = SourceType.WHATSAPP

class RawDataRecord(BaseModel):
    ID: str
    UUID: str
    source_type: str
    sender_phone: str
    is_group_message: bool
    group_name: Optional[str]
    channel_name: Optional[str]
    content_type: str
    content_url: Optional[str]
    raw_text: Optional[str]
    submission_timestamp: str
    processing_status: str
    user_identifier: str
    priority: str

class StorageResponse(BaseModel):
    success: bool
    message: str
    uuid: Optional[str] = None
    file_storage_path: Optional[str] = None
    record_id: Optional[str] = None
    error: Optional[str] = None

class StatusUpdateRequest(BaseModel):
    status: ProcessingStatus

class PaginatedResponse(BaseModel):
    data: List[RawDataRecord]
    count: int
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool

class StorageStats(BaseModel):
    total_records: int
    by_status: Dict[str, int]
    by_content_type: Dict[str, int]
    by_message_type: Dict[str, int]
    storage_info: Dict[str, Any]

# ================================
# API ENDPOINTS (Defined LAST)
# ================================

@app.on_event("startup")
async def startup_event():
    """Initialize database and storage on startup"""
    try:
        init_database()
        logger.info("Storage API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        raise

@app.post("/api/store-data", response_model=StorageResponse)
async def store_data(request: StorageRequest):
    """Store message data in database"""
    try:
        logger.info(f"Storing data for user {request.user_identifier}: {request.content_type}")
        logger.info("StorageRequest Variable Types:")
        logger.info(f"  message_id:      {type(request.message_id)} ({request.message_id})")
        logger.info(f"  chat_jid:        {type(request.chat_jid)} ({request.chat_jid})")
        logger.info(f"  chat_name:       {type(request.chat_name)} ({request.chat_name})")
        logger.info(f"  sender_jid:      {type(request.sender_jid)} ({request.sender_jid})")
        logger.info(f"  sender_name:     {type(request.sender_name)} ({request.sender_name})")
        logger.info(f"  user_identifier: {type(request.user_identifier)} ({request.user_identifier})")
        logger.info(f"  content:         {type(request.content)} ({request.content[:50] if request.content else ''})")
        logger.info(f"  content_type:    {type(request.content_type)} ({request.content_type})")
        logger.info(f"  media_filename:  {type(request.media_filename)} ({request.media_filename})")
        logger.info(f"  media_size:      {type(request.media_size)} ({request.media_size})")        
        logger.info(f"  media_path:      {type(request.media_path)} ({request.media_path})")
        logger.info(f"  is_from_me:      {type(request.is_from_me)} ({request.is_from_me})")
        logger.info(f"  is_group:        {type(request.is_group)} ({request.is_group})")
        logger.info(f"  timestamp:       {type(request.timestamp)} ({request.timestamp})")
        logger.info(f"  priority:        {type(request.priority)} ({request.priority})")
        logger.info(f"  source_type:     {type(request.source_type)} ({request.source_type})")
        
        # Convert to dict and do a thorough check for source_metadata
        data_dict = request.dict()
        
        # Debug the entire received data dictionary
        logger.info(f"DEBUG: Full request data: {data_dict}")
        
        if "source_metadata" in data_dict:
            logger.warning("DEBUG: FOUND AND REMOVING source_metadata from request dictionary")
            del data_dict["source_metadata"]
        
        # Create a simple object with the required attributes for storage
        class SimpleRequest:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
                # Ensure there's no source_metadata attribute
                if hasattr(self, 'source_metadata'):
                    logger.warning("DEBUG: FOUND AND REMOVING source_metadata from SimpleRequest object")
                    delattr(self, 'source_metadata')
        
        req = SimpleRequest(data_dict)
        
        # Store in database
        stored_uuid = store_in_raw_database(req)
        
        # Generate record ID for reference
        record_id = generate_id(request.message_id, request.chat_jid, request.timestamp)
        
        return StorageResponse(
            success=True,
            message="Data stored successfully",
            uuid=stored_uuid,
            record_id=record_id
        )
        
    except Exception as e:
        logger.error(f"Storage error: {e}")
        return StorageResponse(
            success=False,
            message="Failed to store data",
            error=str(e)
        )

@app.post("/api/store-file", response_model=StorageResponse)
async def store_file(
    file: UploadFile = File(...),
    user_identifier: str = Form(...),
    original_filename: str = Form(...),
    description: str = Form(default="")
):
    """Store uploaded file in file storage"""
    try:
        logger.info(f"Storing file for user {user_identifier}: {file.filename}")
        
        # Save file temporarily
        temp_path = await save_uploaded_file(file, user_identifier)
        
        # Store in organized file storage
        stored_path = store_file_in_storage(temp_path, user_identifier, original_filename)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return StorageResponse(
            success=True,
            message="File stored successfully",
            file_storage_path=stored_path
        )
        
    except Exception as e:
        logger.error(f"File storage error: {e}")
        return StorageResponse(
            success=False,
            message="Failed to store file",
            error=str(e)
        )

@app.post("/api/store-complete", response_model=StorageResponse)
async def store_complete_submission(
    request: Request,  # Use Request to access raw form data
    file: Optional[UploadFile] = File(None)
):
    """Store complete submission with optional file"""
    try:        # Parse form data manually
        form_data = await request.form()
        
        # Debug the raw form data
        logger.info("DEBUG: Raw form data received:")
        for key, value in form_data.items():
            if key != "file":  # Skip file to avoid huge log
                logger.info(f"  {key}: {type(value)} = {value}")
        
        # Extract form fields manually
        message_id = str(form_data.get("message_id", ""))
        chat_jid = str(form_data.get("chat_jid", ""))
        chat_name = str(form_data.get("chat_name", ""))
        sender_jid = str(form_data.get("sender_jid", ""))
        sender_name = str(form_data.get("sender_name", ""))
        user_identifier = str(form_data.get("user_identifier", ""))
        content = str(form_data.get("content", ""))        
        content_type = str(form_data.get("content_type", ""))
        media_filename = str(form_data.get("media_filename", ""))
        media_size = int(form_data.get("media_size", "0") or "0")
        
        # Handle boolean fields more robustly
        is_from_me_str = str(form_data.get("is_from_me", "false")).lower()
        is_from_me = is_from_me_str == "true" or is_from_me_str == "t" or is_from_me_str == "1"
        
        is_group_str = str(form_data.get("is_group", "false")).lower()
        is_group = is_group_str == "true" or is_group_str == "t" or is_group_str == "1"
        
        timestamp = str(form_data.get("timestamp", ""))
        priority = str(form_data.get("priority", "normal"))
        source_type = str(form_data.get("source_type", "whatsapp"))
        
        logger.info(f"Storing complete submission for user {user_identifier}")
        
        # Validate required fields
        if not all([message_id, chat_jid, sender_jid, user_identifier, timestamp]):
            raise ValueError("Missing required fields")
        
        file_storage_path = None
        
        # Handle file upload if provided
        if file and file.filename:
            temp_path = await save_uploaded_file(file, user_identifier)
            file_storage_path = store_file_in_storage(
                temp_path, 
                user_identifier, 
                media_filename or file.filename
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Create request object with simplified fields
        class SimpleRequest:
            def __init__(self):
                self.message_id = message_id
                self.chat_jid = chat_jid
                self.chat_name = chat_name
                self.sender_jid = sender_jid
                self.sender_name = sender_name
                self.user_identifier = user_identifier
                self.content = content
                self.content_type = content_type
                self.media_filename = media_filename
                self.media_size = media_size
                self.is_from_me = is_from_me
                self.is_group = is_group
                self.timestamp = timestamp
                self.priority = priority
                self.source_type = source_type
        
        req = SimpleRequest()
        
        # Store in database
        stored_uuid = store_in_raw_database(req, file_storage_path)
        record_id = generate_id(message_id, chat_jid, timestamp)
        
        return StorageResponse(
            success=True,
            message="Complete submission stored successfully",
            uuid=stored_uuid,
            file_storage_path=file_storage_path,
            record_id=record_id
        )
        
    except Exception as e:
        logger.error(f"Complete storage error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return StorageResponse(
            success=False,
            message="Failed to store complete submission",
            error=str(e)
        )

@app.get("/api/records", response_model=PaginatedResponse)
async def get_records(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    content_type: Optional[ContentType] = Query(None, description="Filter by content type"),
    user_identifier: Optional[str] = Query(None, description="Filter by user")
):
    """Get paginated records from database"""
    try:
        offset = (page - 1) * page_size
        
        # Get total count for pagination
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Build WHERE clause
        where_conditions = []
        params = []
        
        if status:
            where_conditions.append("processing_status = ?")
            params.append(status.value)
        
        if content_type:
            where_conditions.append("content_type = ?")
            params.append(content_type.value)
        
        if user_identifier:
            where_conditions.append("user_identifier = ?")
            params.append(user_identifier)
        
        where_clause = " AND ".join(where_conditions)
        where_sql = f"WHERE {where_clause}" if where_clause else ""
        
        # Get total count
        cursor.execute(f"SELECT COUNT(*) FROM raw_data {where_sql}", params)
        total = cursor.fetchone()[0]
        
        # Get paginated data
        cursor.execute(f'''
            SELECT * FROM raw_data {where_sql}
            ORDER BY submission_timestamp DESC 
            LIMIT ? OFFSET ?
        ''', params + [page_size, offset])
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        has_next = (page * page_size) < total
        has_previous = page > 1
        
        return PaginatedResponse(
            data=results,
            count=len(results),
            total=total,
            page=page,
            page_size=page_size,
            has_next=has_next,
            has_previous=has_previous
        )
        
    except Exception as e:
        logger.error(f"Error retrieving records: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/record/{uuid}")
async def get_record_by_uuid(uuid: str):
    """Get specific record by UUID"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM raw_data WHERE UUID = ?", (uuid,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Record not found")
        
        columns = [description[0] for description in cursor.description]
        record = dict(zip(columns, row))
        
        conn.close()
        return record
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.put("/api/record/{uuid}/status")
async def update_record_status(uuid: str, request: StatusUpdateRequest):
    """Update processing status of a record"""
    try:
        result = update_processing_status(uuid, request.status.value)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/latest-pending")
async def get_latest_pending_record():
    """Get the latest pending record for processing"""
    try:
        result = get_latest_pending()
        return result
    except Exception as e:
        logger.error(f"Error getting latest pending: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/file/{user_identifier}/{filename}")
async def get_file(user_identifier: str, filename: str):
    """Retrieve stored file"""
    try:
        file_path = os.path.join(FILE_STORAGE_PATH, user_identifier, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error retrieving file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/{user_identifier}")
async def list_user_files(user_identifier: str):
    """List all files for a user"""
    try:
        user_dir = os.path.join(FILE_STORAGE_PATH, user_identifier)
        
        if not os.path.exists(user_dir):
            return {"files": [], "count": 0}
        
        files = []
        for filename in os.listdir(user_dir):
            file_path = os.path.join(user_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {"files": files, "count": len(files)}
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/record/{uuid}")
async def delete_record(uuid: str, delete_file: bool = Query(False, description="Also delete associated file")):
    """Delete a record and optionally its associated file"""
    try:
        # First get the record to find associated file
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT content_url FROM raw_data WHERE UUID = ?", (uuid,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Record not found")
        
        file_path = row[0]
        
        # Delete from database
        cursor.execute("DELETE FROM raw_data WHERE UUID = ?", (uuid,))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        
        conn.commit()
        conn.close()
        
        # Delete file if requested and exists
        if delete_file and file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        
        return {"message": "Record deleted successfully", "uuid": uuid}
        
    except Exception as e:
        logger.error(f"Error deleting record: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=StorageStats)
async def get_storage_stats():
    """Get comprehensive storage statistics"""
    try:
        # Get database stats
        db_stats = get_stats()
        
        # Get storage info
        storage_info = get_storage_info()
        
        return StorageStats(
            total_records=db_stats["total_records"],
            by_status=db_stats["by_status"],
            by_content_type=db_stats["by_content_type"],
            by_message_type=db_stats["by_message_type"],
            storage_info=storage_info
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cleanup")
async def cleanup_orphaned_files(background_tasks: BackgroundTasks):
   """Clean up orphaned files that aren't referenced in database"""
   
   def cleanup_task():
       try:
           conn = sqlite3.connect(DB_PATH)
           cursor = conn.cursor()
           
           # Get all file paths from database
           cursor.execute("SELECT content_url FROM raw_data WHERE content_url IS NOT NULL")
           db_files = {row[0] for row in cursor.fetchall() if row[0]}
           conn.close()
           
           # Find all files in storage
           orphaned_files = []
           for root, dirs, files in os.walk(FILE_STORAGE_PATH):
               for file in files:
                   file_path = os.path.join(root, file)
                   if file_path not in db_files:
                       orphaned_files.append(file_path)
           
           # Log orphaned files (could delete them here if needed)
           logger.info(f"Found {len(orphaned_files)} orphaned files")
           for file_path in orphaned_files:
               logger.info(f"Orphaned file: {file_path}")
           
       except Exception as e:
           logger.error(f"Cleanup task error: {e}")
   
   background_tasks.add_task(cleanup_task)
   return {"message": "Cleanup task started in background"}

@app.get("/api/health")
async def health_check():
   """Health check endpoint"""
   try:
       # Test database connection
       conn = sqlite3.connect(DB_PATH)
       cursor = conn.cursor()
       cursor.execute("SELECT 1")
       conn.close()
       
       storage_info = get_storage_info()
       
       return {
           "status": "healthy",
           "timestamp": datetime.now(timezone.utc).isoformat(),
           "database": {
               "accessible": True,
               "path": DB_PATH,
               "exists": storage_info["database_exists"],
               "size_bytes": storage_info["database_size_bytes"]
           },
           "file_storage": {
               "accessible": True,
               "path": FILE_STORAGE_PATH,
               "exists": storage_info["file_storage_exists"],
               "size_bytes": storage_info["file_storage_size_bytes"],
               "file_count": storage_info["file_count"]
           }
       }
       
   except Exception as e:
       return {
           "status": "unhealthy",
           "error": str(e),
           "timestamp": datetime.now(timezone.utc).isoformat()
       }

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8002)