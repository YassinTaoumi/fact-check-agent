# processing_api.py - File Processing & Validation API
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
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
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import socket


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Content Processing & Validation API",
    description="API for processing and validating content files before extraction",
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

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff"}
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv", "video/webm", "video/flv"}
SUPPORTED_AUDIO_TYPES = {"audio/ogg", "audio/mp3", "audio/wav", "audio/m4a", "audio/flac", "audio/aac"}
SUPPORTED_DOCUMENT_TYPES = {"application/pdf", "text/plain", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

UPLOAD_DIRECTORY = "uploads"
PROCESSED_DIRECTORY = "processed"

# Ensure directories exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)

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

# Custom Exceptions
class ValidationError(Exception):
    pass

class FileSizeError(ValidationError):
    pass

class UnsupportedFormatError(ValidationError):
    pass

class MaliciousContentError(ValidationError):
    pass

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
    source_type: Optional[SourceType] = SourceType.WHATSAPP
    priority: Optional[str] = "normal"

class ValidationResult(BaseModel):
    is_valid: bool
    file_path: Optional[str] = None
    detected_mime_type: Optional[str] = None
    actual_file_size: Optional[int] = None
    validation_errors: List[str] = Field(default_factory=list)
    security_warnings: List[str] = Field(default_factory=list)
    is_group: Optional[bool] = None
    group_name: Optional[str] = None
    is_channel: Optional[bool] = None
    channel_name: Optional[str] = None

class ProcessingResponse(BaseModel):
    success: bool
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    validation_result: Optional[Dict[str, Any]] = None
    processed_file_path: Optional[str] = None
    content_type: ContentType
    ready_for_extraction: bool = False
    error: Optional[str] = None

class FileInfo(BaseModel):
    filename: str
    size: int
    mime_type: str
    content_type: ContentType
    validation_status: ProcessingStatus

# Helper Functions
def determine_message_context(chat_jid: str, chat_name: str, is_group: bool) -> Dict[str, Any]:
    """Determine if message is from group or channel and extract names"""
    context = {
        "is_group": False,
        "group_name": None,
        "is_channel": False,
        "channel_name": None
    }
    
    if is_group and chat_jid and chat_name:
        # Check if it's a WhatsApp channel
        if '@newsletter' in chat_jid:
            context["is_channel"] = True
            context["channel_name"] = chat_name
        elif '@g.us' in chat_jid:
            # Regular WhatsApp group
            context["is_group"] = True
            context["group_name"] = chat_name
    
    return context

# Validation Functions
def determine_content_type_from_text(content: str) -> ContentType:
    """Determine content type based on simple, reliable URL detection"""
    if not content:
        return ContentType.TEXT
    
    # Use simple URL detection without external dependencies
    if _simple_url_detection(content):
        logger.info(f"Content classified as 'link' - found URL patterns in: {content[:100]}...")
        return ContentType.LINK
    
    return ContentType.TEXT

def _simple_url_detection(text: str) -> bool:
    """
    Simple, reliable URL detection without regex or external dependencies
    Returns True if text contains what appears to be a valid URL
    """
    if not text:
        return False
    
    # Split text into tokens using common delimiters
    delimiters = [' ', '\t', '\n', '\r', ',', ';', '(', ')', '[', ']', '{', '}', 
                  '"', "'", '<', '>', '|', '\\']
    
    tokens = [text]
    for delimiter in delimiters:
        new_tokens = []
        for token in tokens:
            new_tokens.extend(token.split(delimiter))
        tokens = new_tokens
    
    # Check each token for URL-like patterns
    for token in tokens:
        if not token:
            continue
            
        # Clean trailing punctuation
        cleaned = token.rstrip('.,!?;:"\'')
        if not cleaned:
            continue
        
        # Check for protocol URLs (most reliable)
        if any(cleaned.lower().startswith(proto) for proto in 
               ['http://', 'https://', 'ftp://', 'ftps://']):
            logger.info(f"Found protocol URL: {cleaned}")
            return True
        
        # Check for www. domains
        if cleaned.lower().startswith('www.') and '.' in cleaned[4:]:
            logger.info(f"Found www domain: {cleaned}")
            return True
        
        # Check for domain-like patterns (basic validation)
        if ('.' in cleaned and 
            not cleaned.startswith('.') and 
            not cleaned.endswith('.') and
            len(cleaned.split('.')) >= 2):
            
            parts = cleaned.split('.')
            domain_part = parts[0]
            tld_part = parts[-1]
            
            # Basic validation
            if (len(domain_part) > 1 and 
                len(tld_part) >= 2 and 
                len(tld_part) <= 6 and
                tld_part.isalpha() and
                not cleaned.replace('.', '').isdigit()):  # Not just numbers
                
                # Check against common false positives
                if not any(fp in cleaned.lower() for fp in ['file.', '.txt', '.jpg', '.pdf', '.doc', '.png', '.mp4']):
                    logger.info(f"Found domain-like URL: {cleaned}")
                    return True
    
    return False

def validate_text_content(content: str) -> List[str]:
    """Validate text content for malicious patterns"""
    warnings = []
    
    if not content:
        return warnings
    
    # Basic text-based malicious patterns
    suspicious_patterns = [
        "<script", "javascript:", "<?php", "eval(", 
        "document.cookie", "window.location", "exec(",
        "system(", "shell_exec", "file_get_contents",
        "base64_decode", "eval(base64"
    ]
    
    content_lower = content.lower()
    for pattern in suspicious_patterns:
        if pattern in content_lower:
            warnings.append(f"Potentially malicious pattern detected: {pattern}")
    
    # Check for extremely long content (possible DoS)
    if len(content) > 1000000:  # 1MB of text
        warnings.append("Extremely long text content detected")
    
    return warnings

def validate_file_type(file_path: str, expected_type: str) -> tuple[bool, str, List[str]]:
    """Validate file type and check for security issues"""
    errors = []
    warnings = []
    
    try:
        # Check file exists
        if not os.path.exists(file_path):
            errors.append("File does not exist")
            return False, "", errors
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            errors.append(f"File size {file_size} exceeds maximum {MAX_FILE_SIZE}")
            return False, "", errors
        
        # Get MIME type
        mime_type = get_file_mime_type(file_path)
        
        # Validate against expected type
        if expected_type == "image" and mime_type not in SUPPORTED_IMAGE_TYPES:
            errors.append(f"Unsupported image type: {mime_type}")
        elif expected_type == "video" and mime_type not in SUPPORTED_VIDEO_TYPES:
            errors.append(f"Unsupported video type: {mime_type}")
        elif expected_type == "audio" and mime_type not in SUPPORTED_AUDIO_TYPES:
            errors.append(f"Unsupported audio type: {mime_type}")
        elif expected_type == "document" and mime_type not in SUPPORTED_DOCUMENT_TYPES:
            errors.append(f"Unsupported document type: {mime_type}")
        
        # Basic security checks
        if file_size == 0:
            warnings.append("Empty file detected")
        
        # Check for suspicious file signatures
        with open(file_path, 'rb') as f:
            file_header = f.read(16)
            
        # Check for executable signatures
        exe_signatures = [
            b'MZ',  # Windows executable
            b'\x7fELF',  # Linux executable
            b'\xfe\xed\xfa',  # Mach-O executable
        ]
        
        for sig in exe_signatures:
            if file_header.startswith(sig):
                warnings.append("Executable file signature detected")
                break
        
        return len(errors) == 0, mime_type, errors + warnings
        
    except Exception as e:
        errors.append(f"File validation error: {str(e)}")
        return False, "", errors

# File Processing Functions
def get_file_mime_type(file_path: str) -> str:
    """Get MIME type of a file"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        # Fallback to reading file signature
        try:
            with open(file_path, 'rb') as f:
                file_signature = f.read(16)
                if file_signature.startswith(b'\xFF\xD8\xFF'):
                    return "image/jpeg"
                elif file_signature.startswith(b'\x89PNG'):
                    return "image/png"
                elif file_signature.startswith(b'GIF8'):
                    return "image/gif"
                elif file_signature.startswith(b'RIFF') and b'WEBP' in file_signature:
                    return "image/webp"
                elif file_signature.startswith(b'%PDF'):
                    return "application/pdf"
                elif file_signature.startswith(b'ftyp'):
                    return "video/mp4"
        except Exception as e:
            logger.warning(f"Could not read file signature for {file_path}: {e}")
    
    return mime_type or "application/octet-stream"

def determine_content_type_from_mime(mime_type: str) -> ContentType:
    """Determine ContentType enum from MIME type"""
    if mime_type in SUPPORTED_IMAGE_TYPES:
        return ContentType.IMAGE
    elif mime_type in SUPPORTED_VIDEO_TYPES:
        return ContentType.VIDEO
    elif mime_type in SUPPORTED_AUDIO_TYPES:
        return ContentType.AUDIO
    elif mime_type == "application/pdf":
        return ContentType.PDF
    elif mime_type in SUPPORTED_DOCUMENT_TYPES:
        return ContentType.DOCUMENT
    else:
        return ContentType.DOCUMENT

async def process_uploaded_file(file: UploadFile, user_identifier: str) -> tuple[str, int]:
    """Process uploaded file and return local path and size"""
    # Create user-specific upload directory
    user_upload_dir = os.path.join(UPLOAD_DIRECTORY, user_identifier)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(file.filename).suffix if file.filename else ""
    unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
    file_path = os.path.join(user_upload_dir, unique_filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return file_path, len(content)

# API Endpoints
@app.post("/api/process-text", response_model=ProcessingResponse)
async def process_text_content(request: ProcessingRequest):
    """Process text content for validation and reliable link detection"""
    
    try:
        logger.info(f"Processing text content from {request.user_identifier}")
        logger.info("ProcessingRequest Variable Types:")
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
        logger.info(f"  source_type:     {type(request.source_type)} ({request.source_type})")
        logger.info(f"  priority:        {type(request.priority)} ({request.priority})")
        
        # Determine message context (group/channel)
        message_context = determine_message_context(
            request.chat_jid, 
            request.chat_name, 
            request.is_group
        )
        
        # Detect and analyze content type using simple, reliable URL detection
        detected_content_type = determine_content_type_from_text(request.content)
        
        # Validate text content for security issues
        security_warnings = validate_text_content(request.content)
        
        # Create enhanced validation result
        validation_result = ValidationResult(
            is_valid=True,
            validation_errors=[],
            security_warnings=security_warnings,
            **message_context  # Include group/channel info
        )
        
        # Add basic link information to validation result
        validation_data = validation_result.dict()
        has_links = detected_content_type == ContentType.LINK
        if has_links:
            validation_data.update({
                'has_links': True,
                'detected_as_link': True
            })
        
        message = f"Content processed successfully - detected as {detected_content_type.value}"
        if has_links:
            message += " (contains URL patterns)"
        
        # Add context info to message
        if message_context["is_group"]:
            message += f" from group: {message_context['group_name']}"
        elif message_context["is_channel"]:
            message += f" from channel: {message_context['channel_name']}"
        
        return ProcessingResponse(
            success=True,
            message=message,
            validation_result=validation_data,
            content_type=detected_content_type,
            ready_for_extraction=True
        )
        
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        return ProcessingResponse(
            success=False,
            message="Text processing failed",
            content_type=request.content_type,
            ready_for_extraction=False,
            error=str(e)
        )

@app.post("/api/process-file", response_model=ProcessingResponse)
async def process_file_upload(
    file: UploadFile = File(...),
    user_identifier: str = Form(...),
    message_id: str = Form(...),
    chat_jid: str = Form(...),
    chat_name: str = Form(default=""),
    sender_jid: str = Form(...),
    sender_name: str = Form(...),
    timestamp: str = Form(...),
    content: str = Form(default=""),
    content_type: str = Form(default="document"),
    is_from_me: bool = Form(default=False),
    is_group: bool = Form(default=False),
    source_type: str = Form(default="whatsapp"),
    priority: str = Form(default="normal")
):
    """Process uploaded file for validation"""
    
    try:
        
        logger.info(f"Processing file upload from {user_identifier}: {file.filename}")
        
        # Determine message context (group/channel)
        message_context = determine_message_context(chat_jid, chat_name, is_group)
        
        # Process the uploaded file
        file_path, file_size = await process_uploaded_file(file, user_identifier)
        
        # Validate file
        mime_type = get_file_mime_type(file_path)
        detected_content_type = determine_content_type_from_mime(mime_type)
        
        # Validate file type and security
        is_valid, detected_mime, validation_messages = validate_file_type(
            file_path, 
            detected_content_type.value
        )
        
        # Create validation result
        validation_result = ValidationResult(
            is_valid=is_valid,
            file_path=file_path,
            detected_mime_type=detected_mime,
            actual_file_size=file_size,
            validation_errors=[msg for msg in validation_messages if "error" in msg.lower()],
            security_warnings=[msg for msg in validation_messages if "warning" in msg.lower()],
            **message_context  # Include group/channel info
        )
        
        message = f"File processed successfully: {file.filename}"
        if message_context["is_group"]:
            message += f" from group: {message_context['group_name']}"
        elif message_context["is_channel"]:
            message += f" from channel: {message_context['channel_name']}"
        
        return ProcessingResponse(
            success=True,
            message=message,
            validation_result=validation_result.dict(),
            processed_file_path=file_path,
            content_type=detected_content_type,
            ready_for_extraction=is_valid
        )
        
    except Exception as e:
        logger.error(f"File processing error: {e}")
        return ProcessingResponse(
            success=False,
            message="File processing failed",
            content_type=ContentType.DOCUMENT,
            ready_for_extraction=False,
            error=str(e)
        )

@app.get("/api/validate-file/{file_path:path}")
async def validate_existing_file(file_path: str):
    """Validate an existing file"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        mime_type = get_file_mime_type(file_path)
        content_type = determine_content_type_from_mime(mime_type)
        
        is_valid, detected_mime, validation_messages = validate_file_type(
            file_path, 
            content_type.value
        )
        
        return {
            "file_path": file_path,
            "is_valid": is_valid,
            "mime_type": detected_mime,
            "content_type": content_type.value,
            "file_size": os.path.getsize(file_path),
            "validation_messages": validation_messages
        }
        
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "image_types": list(SUPPORTED_IMAGE_TYPES),
        "video_types": list(SUPPORTED_VIDEO_TYPES),
        "audio_types": list(SUPPORTED_AUDIO_TYPES),
        "document_types": list(SUPPORTED_DOCUMENT_TYPES),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_file_size_bytes": MAX_FILE_SIZE,
        "content_types": [ct.value for ct in ContentType],
        "source_types": [st.value for st in SourceType]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "storage": {
            "upload_directory": os.path.exists(UPLOAD_DIRECTORY),
            "processed_directory": os.path.exists(PROCESSED_DIRECTORY)
        },
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
        },
        "features": {
            "link_detection": True,
            "group_channel_detection": True,
            "file_validation": True,
            "security_scanning": True
        }
    }

@app.get("/api/stats")
async def get_processing_stats():
    """Get processing statistics"""
    try:
        upload_files = len([f for f in os.listdir(UPLOAD_DIRECTORY) if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, f))])
        processed_files = len([f for f in os.listdir(PROCESSED_DIRECTORY) if os.path.isfile(os.path.join(PROCESSED_DIRECTORY, f))])
        
        # Get user-specific stats
        user_dirs = [d for d in os.listdir(UPLOAD_DIRECTORY) if os.path.isdir(os.path.join(UPLOAD_DIRECTORY, d))]
        user_file_counts = {}
        for user_dir in user_dirs:
            user_path = os.path.join(UPLOAD_DIRECTORY, user_dir)
            user_file_counts[user_dir] = len([f for f in os.listdir(user_path) if os.path.isfile(os.path.join(user_path, f))])
        
        return {
            "files_in_upload": upload_files,
            "files_processed": processed_files,
            "user_file_counts": user_file_counts,
            "supported_formats_count": {
                "images": len(SUPPORTED_IMAGE_TYPES),
                "videos": len(SUPPORTED_VIDEO_TYPES),
                "audio": len(SUPPORTED_AUDIO_TYPES),
                "documents": len(SUPPORTED_DOCUMENT_TYPES)
            },
            "processing_features": {
                "link_detection_enabled": True,
                "group_channel_detection_enabled": True,
                "file_validation_enabled": True,
                "security_scanning_enabled": True
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/cleanup")
async def cleanup_temp_files(background_tasks: BackgroundTasks):
    """Clean up temporary processing files"""
    
    def cleanup_task():
        try:
            # Clean up files older than 1 hour in upload directory
            current_time = datetime.now().timestamp()
            deleted_count = 0
            
            for root, dirs, files in os.walk(UPLOAD_DIRECTORY):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > 3600:  # 1 hour
                            os.remove(file_path)
                            deleted_count += 1
                            logger.info(f"Deleted old temp file: {file_path}")
            
            logger.info(f"Cleanup completed: deleted {deleted_count} old files")
            
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
    
    background_tasks.add_task(cleanup_task)
    return {"message": "Cleanup task started in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)