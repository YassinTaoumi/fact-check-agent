# main.py - FastAPI application orchestrating combined processing and storage API
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fact-Checking Orchestration API", 
    description="Main API that orchestrates the combined processing and storage microservice",
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

# Configuration for API endpoints
COMBINED_API_URL = os.getenv("COMBINED_API_URL", "http://localhost:8001")

# Timeout for API calls
API_TIMEOUT = 30.0

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
async def call_combined_api(endpoint: str, method: str = "GET", data: dict = None, files: dict = None) -> dict:
    """Make API call to Combined Processing & Storage API"""
    url = f"{COMBINED_API_URL}{endpoint}"
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            if method == "POST":
                if files:
                    response = await client.post(url, data=data, files=files)
                else:
                    response = await client.post(url, json=data)
            elif method == "GET":
                response = await client.get(url, params=data)
            elif method == "PUT":
                response = await client.put(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
    except httpx.TimeoutException:
        logger.error(f"Timeout calling Combined API: {url}")
        raise HTTPException(status_code=504, detail="Combined API timeout")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling Combined API: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Combined API error: {e.response.text}")
    except Exception as e:
        logger.error(f"Error calling Combined API: {e}")
        raise HTTPException(status_code=500, detail=f"Combined API error: {str(e)}")

def determine_message_context(chat_jid: str, chat_name: str, is_group: bool) -> Dict[str, Any]:
    """Determine message context and extract group/channel information"""
    context = {
        "is_group_message": is_group,
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
            logger.info(f"Detected WhatsApp channel: {chat_name}")
        elif '@g.us' in chat_jid:
            # Regular WhatsApp group
            context["is_group"] = True
            context["group_name"] = chat_name
            logger.info(f"Detected WhatsApp group: {chat_name}")
    
    return context

# Main API Endpoints
@app.post("/api/process-message", response_model=ProcessingResponse)
async def process_message(req: APIProcessingRequest):
    """Main endpoint for processing incoming messages - using the combined processing and storage API"""
    
    try:
        logger.info(f"Processing message from {req.user_identifier}: {req.content_type}")
        logger.info(f"DEBUG: Request data: {req.dict()}")
        
        message_context = determine_message_context(req.chat_jid, req.chat_name, req.is_group)
        
        # Process and store the content using the combined API
        combined_result = await call_combined_api("/api/process-message", "POST", req.dict())
        logger.info(f"DEBUG: Combined API response: {combined_result}")
        
        if not combined_result.get("success", False):
            return ProcessingResponse(
                success=False,
                message="Processing and storage failed",
                error=combined_result.get("error", "Unknown error"),
                should_retry=True if combined_result.get("should_retry", False) else False
            )
        
        # Create success message with context
        success_message = "Message processed and stored successfully"
        if message_context.get("is_group"):
            success_message += f" from group: {message_context.get('group_name')}"
        elif message_context.get("is_channel"):
            success_message += f" from channel: {message_context.get('channel_name')}"
        
        return ProcessingResponse(
            success=True,
            message=success_message,
            stored_id=combined_result.get("stored_id") or combined_result.get("uuid"),
            file_storage_path=combined_result.get("file_storage_path"),
            processing_job_id=combined_result.get("job_id"),
            validation_result=combined_result.get("validation_result"),
            should_retry=False
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Orchestration error: {e}")
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
    """Process file upload through both processing and storage APIs"""
    
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
        logger.info(f"  content:         {type(content)} ({content[:50] if content else ''})")
        logger.info(f"  content_type:    {type(content_type)} ({content_type})")
        logger.info(f"  media_filename:  {type(media_filename)} ({media_filename})")
        logger.info(f"  media_size:      {type(media_size)} ({media_size})")
        logger.info(f"  is_from_me:      {type(is_from_me)} ({is_from_me})")
        logger.info(f"  is_group:        {type(is_group)} ({is_group})")
        logger.info(f"  source_type:     {type(source_type)} ({source_type})")
        logger.info(f"  priority:        {type(priority)} ({priority})")
        
        # Determine message context (group/channel info)
        message_context = determine_message_context(chat_jid, chat_name, is_group)
        
        # Step 1: Process file using Processing API
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer for storage API
        
        files = {"file": (file.filename, file_content, file.content_type)}
        processing_data = {
            "user_identifier": user_identifier,
            "message_id": message_id,
            "chat_jid": chat_jid,
            "chat_name": chat_name,
            "sender_jid": sender_jid,
            "sender_name": sender_name,
            "timestamp": timestamp,
            "content": content,
            "content_type": content_type,
            "is_from_me": str(is_from_me).lower(),
            "is_group": str(is_group).lower(),
            "source_type": source_type,
            "priority": priority
        }
        
        processing_result = await call_processing_api("/api/process-file", "POST", processing_data, files)
        
        if not processing_result.get("success", False):
            return ProcessingResponse(
                success=False,
                message="File processing failed",
                error=processing_result.get("error", "Unknown processing error"),
                should_retry=False
            )

        # Step 2: Store complete submission using Storage API
        await file.seek(0)  # Reset file pointer again
        
        storage_form_data = {
            "message_id": message_id,
            "chat_jid": chat_jid,
            "chat_name": chat_name,
            "sender_jid": sender_jid,
            "sender_name": sender_name,
            "user_identifier": user_identifier,            "content": content,
            "content_type": content_type,
            "media_filename": file.filename,
            "media_size": str(len(file_content)),
            "is_from_me": "false",  # Always false as string to avoid type confusion
            "is_group": "true" if is_group else "false",  # Use string values to avoid type confusion
            "timestamp": timestamp,
            "priority": priority,
            "source_type": source_type
        }
        
        files = {"file": (file.filename, file_content, file.content_type)}
        
        storage_result = await call_storage_api("/api/store-complete", "POST", storage_form_data, files)
        
        if not storage_result.get("success", False):
            return ProcessingResponse(
                success=False,
                message="Storage failed",
                error=storage_result.get("error", "Unknown storage error"),
                should_retry=True
            )
        
        # Create success message with context
        success_message = "File processed and stored successfully"
        if message_context["is_group"]:
            success_message += f" from group: {message_context['group_name']}"
        elif message_context["is_channel"]:
            success_message += f" from channel: {message_context['channel_name']}"
        
        return ProcessingResponse(
            success=True,
            message=success_message,
            stored_id=storage_result.get("uuid"),
            file_storage_path=storage_result.get("file_storage_path"),
            processing_job_id=processing_result.get("job_id"),
            validation_result=processing_result.get("validation_result"),
            should_retry=False
        )
        
    except Exception as e:
        logger.error(f"File processing orchestration error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return ProcessingResponse(
            success=False,
            message="File processing failed",
            error=str(e),
            should_retry=True
        )

# Proxy endpoints to Storage API
@app.get("/api/raw-data")
async def get_raw_data_endpoint(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    user_identifier: Optional[str] = None,
    is_group: Optional[bool] = None,
    is_channel: Optional[bool] = None
):
    """Get raw data records from storage with group/channel filtering"""
    params = {
        "page": page,
        "page_size": page_size
    }
    if status:
        params["status"] = status
    if content_type:
        params["content_type"] = content_type
    if user_identifier:
        params["user_identifier"] = user_identifier
    if is_group is not None:
        params["is_group"] = is_group
    if is_channel is not None:
        params["is_channel"] = is_channel
    
    return await call_storage_api("/api/records", "GET", params)

@app.get("/api/latest-pending")
async def get_latest_pending_endpoint():
    """Get the latest pending record for extraction processing"""
    return await call_storage_api("/api/latest-pending", "GET")

@app.put("/api/update-status/{uuid}")
async def update_processing_status_endpoint(uuid: str, status: str):
    """Update processing status of a record"""
    data = {"status": status}
    return await call_storage_api(f"/api/record/{uuid}/status", "PUT", data)

@app.get("/api/record/{uuid}")
async def get_record_endpoint(uuid: str):
    """Get specific record by UUID"""
    return await call_storage_api(f"/api/record/{uuid}", "GET")

@app.delete("/api/record/{uuid}")
async def delete_record_endpoint(uuid: str, delete_file: bool = False):
    """Delete a record and optionally its file"""
    params = {"delete_file": delete_file}
    return await call_storage_api(f"/api/record/{uuid}", "DELETE", params)

@app.get("/api/files/{user_identifier}")
async def list_user_files_endpoint(user_identifier: str):
    """List all files for a user"""
    return await call_storage_api(f"/api/files/{user_identifier}", "GET")

# Proxy endpoints to Processing API
@app.get("/api/supported-formats")
async def get_supported_formats_endpoint():
    """Get supported file formats from processing API"""
    return await call_processing_api("/api/supported-formats", "GET")

@app.get("/api/validate-file/{file_path:path}")
async def validate_file_endpoint(file_path: str):
    """Validate an existing file"""
    return await call_processing_api(f"/api/validate-file/{file_path}", "GET")

# New endpoints for group/channel data
@app.get("/api/groups")
async def get_groups_endpoint():
    """Get all groups that have sent messages"""
    try:
        # This would need to be implemented in storage API
        # For now, return a placeholder
        return {
            "message": "Groups endpoint - to be implemented in storage API",
            "groups": []
        }
    except Exception as e:
        logger.error(f"Error getting groups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/channels")
async def get_channels_endpoint():
    """Get all channels that have sent messages"""
    try:
        # This would need to be implemented in storage API
        # For now, return a placeholder
        return {
            "message": "Channels endpoint - to be implemented in storage API",
            "channels": []
        }
    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health and Stats endpoints
@app.get("/api/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check for all services"""
    
    services = {}
    overall_health = True
    
    # Check Processing API
    try:
        processing_health = await call_processing_api("/api/health", "GET")
        services["processing_api"] = {
            "status": "healthy",
            "url": PROCESSING_API_URL,
            "response": processing_health
        }
    except Exception as e:
        services["processing_api"] = {
            "status": "unhealthy",
            "url": PROCESSING_API_URL,
            "error": str(e)
        }
        overall_health = False
    
    # Check Storage API
    try:
        storage_health = await call_storage_api("/api/health", "GET")
        services["storage_api"] = {
            "status": "healthy",
            "url": STORAGE_API_URL,
            "response": storage_health
        }
    except Exception as e:
        services["storage_api"] = {
            "status": "unhealthy",
            "url": STORAGE_API_URL,
            "error": str(e)
        }
        overall_health = False
    
    return HealthStatus(
        status="healthy" if overall_health else "degraded",
        timestamp=datetime.now().isoformat(),
        services=services,
        overall_health=overall_health
    )

@app.get("/api/stats")
async def get_stats_endpoint():
    """Get comprehensive statistics from all services"""
    
    try:
        # Get stats from both APIs
        processing_stats = await call_processing_api("/api/stats", "GET")
        storage_stats = await call_storage_api("/api/stats", "GET")
        
        return {
            "processing": processing_stats,
            "storage": storage_stats,
            "combined": {
                "total_files_processed": storage_stats.get("storage_info", {}).get("file_count", 0),
                "total_records": storage_stats.get("total_records", 0),
                "by_message_type": storage_stats.get("by_message_type", {}),
                "features_enabled": {
                    "group_detection": True,
                    "channel_detection": True,
                    "link_detection": True,
                    "file_validation": True
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting combined stats: {e}")
        return {"error": str(e)}

@app.get("/api/system-status")
async def get_system_status():
    """Get overall system status and configuration"""
    return {
        "service": "fact-checking-orchestration-api",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "processing_api_url": PROCESSING_API_URL,
            "storage_api_url": STORAGE_API_URL,
            "api_timeout": API_TIMEOUT
        },
        "endpoints": {
            "processing": f"{PROCESSING_API_URL}/docs",
            "storage": f"{STORAGE_API_URL}/docs",
            "orchestration": "/docs"
        },
        "features": {
            "group_detection": True,
            "channel_detection": True,
            "link_detection": True,
            "file_processing": True,
            "content_validation": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)