import yt_dlp
from faster_whisper import WhisperModel
import os
import json
import re
from urllib.parse import urlparse, parse_qs
import torch
import logging
from typing import Dict, Any, Optional
import subprocess
import sys

# Add path to parent directory for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_normalizer import normalize_result, get_text_preview


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_youtube_url(url: str) -> bool:
    """
    Check if a URL is a YouTube URL.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if it's a YouTube URL, False otherwise
    """
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+',
        r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=[\w-]+'
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url, re.IGNORECASE):
            return True
    return False

def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> Optional[str]:
    """
    Extract audio from a local video file using FFmpeg.
    
    Args:
        video_path (str): Path to the video file
        output_audio_path (str): Output path for audio file (optional)
        
    Returns:
        str: Path to extracted audio file, or None if failed
    """
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        if output_audio_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_audio_path = f"{base_name}_audio.mp3"
        
        logger.info(f"Extracting audio from: {video_path}")
        
        # Use FFmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'mp3',  # Audio codec
            '-ab', '192k',  # Audio bitrate
            '-ar', '44100',  # Audio sample rate
            '-y',  # Overwrite output file
            output_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Audio extracted to: {output_audio_path}")
            return output_audio_path
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting audio from video: {e}")
        return None

def download_youtube_audio(youtube_url: str, output_filename_base: str = "youtube_audio") -> Optional[str]:
    """
    Downloads the audio from a YouTube link using yt-dlp and saves it as an MP3.

    Args:
        youtube_url (str): The URL of the YouTube video.
        output_filename_base (str): The desired base name for the output MP3 file

    Returns:
        str: The path to the downloaded audio file, or None if an error occurred.
    """
    # yt-dlp will automatically add the .mp3 extension due to the postprocessor
    output_path_template = output_filename_base

    ydl_opts = {
        'format': 'bestaudio/best',  # Select the best audio quality
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',  # Audio quality (e.g., 192kbps)
        }],
        'outtmpl': output_path_template,  # Output file template (yt-dlp adds .mp3)
        'noplaylist': True,  # Only download the single video, not a playlist
        'quiet': True,  # Suppress most output
    }

    # Construct the expected final filename
    final_output_path = f"{output_filename_base}.mp3"

    try:
        logger.info(f"Downloading audio from YouTube: {youtube_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        logger.info(f"Audio downloaded to: {final_output_path}")
        return final_output_path
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None

def transcribe_audio_fast_whisper(audio_file_path: str, model_size: str = "large-v3", device: str = "cuda") -> Optional[str]:
    """
    Transcribes an audio file using the fast-whisper package.

    Args:
        audio_file_path (str): The path to the audio file.
        model_size (str): The size of the Whisper model to use.
        device (str): The device to run the model on ("cuda" for GPU, "cpu" for CPU).

    Returns:
        str: The transcribed text, or None if failed.
    """
    try:
        # Check if CUDA is available and adjust device accordingly
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
            
        # Determine compute type based on device
        compute_type = "float16" if device == "cuda" and torch.cuda.is_available() else "int8"
        logger.info(f"Loading Whisper model '{model_size}' on {device} with compute type {compute_type}...")
        
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Model loaded. Transcribing...")

        segments, info = model.transcribe(audio_file_path, beam_size=5)

        transcribed_text = ""
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        for segment in segments:
            transcribed_text += segment.text.strip() + " "
            logger.debug(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        return transcribed_text.strip()
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

def process_database_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a database record and transcribe video/audio content.
    
    Args:
        record: Database record with fields from raw_data table:
                ID, UUID, source_type, sender_phone, is_group_message, 
                group_name, channel_name, chat_jid, content_type, content_url,
                raw_text, submission_timestamp, processing_status, user_identifier,
                priority, metadata
        
    Returns:
        Updated record with transcription results
    """
    try:
        content_type = record.get('content_type', '').lower()
        audio_file_path = None
        cleanup_files = []
        record_id = record.get('UUID') or record.get('ID', 'unknown')
        
        logger.info(f"Processing record {record_id} with content_type: {content_type}")
        
        if content_type == 'video':
            # Handle local video file - check content_url first, then raw_text
            video_path = record.get('content_url') or record.get('raw_text')
            if not video_path or not os.path.exists(video_path):
                return {
                    **record,
                    "transcription_status": "failed",
                    "transcription_error": f"Video file not found: {video_path}",
                    "transcription": None,
                    "transcription_word_count": 0
                }
            
            # Extract audio from video
            output_audio_path = f"temp_audio_{record_id}.mp3"
            audio_file_path = extract_audio_from_video(video_path, output_audio_path)
            if audio_file_path:
                cleanup_files.append(audio_file_path)
            
        elif content_type == 'link':
            # Handle YouTube links - check raw_text first, then content_url
            url = record.get('raw_text', '').strip()
            if not url:
                url = record.get('content_url', '').strip()
            
            if not url:
                return {
                    **record,
                    "transcription_status": "failed",
                    "transcription_error": "No URL found in raw_text or content_url",
                    "transcription": None,
                    "transcription_word_count": 0
                }
            
            if not is_youtube_url(url):
                return {
                    **record,
                    "transcription_status": "skipped",
                    "transcription_error": "Not a YouTube URL",
                    "transcription": None,
                    "transcription_word_count": 0
                }
            
            # Download audio from YouTube
            output_base = f"youtube_audio_{record_id}"
            audio_file_path = download_youtube_audio(url, output_base)
            if audio_file_path:
                cleanup_files.append(audio_file_path)
                
        elif content_type == 'audio':
            # Handle audio files directly
            audio_path = record.get('content_url') or record.get('raw_text')
            if not audio_path or not os.path.exists(audio_path):
                return {
                    **record,
                    "transcription_status": "failed",
                    "transcription_error": f"Audio file not found: {audio_path}",
                    "transcription": None,
                    "transcription_word_count": 0
                }
            audio_file_path = audio_path
            # Don't add to cleanup_files since it's an original file
                
        else:
            return {
                **record,
                "transcription_status": "skipped",
                "transcription_error": f"Unsupported content_type: {content_type}",
                "transcription": None,
                "transcription_word_count": 0
            }
        
        # Check if we have an audio file to transcribe
        if not audio_file_path or not os.path.exists(audio_file_path):
            return {
                **record,
                "transcription_status": "failed",
                "transcription_error": "Failed to extract/download audio",
                "transcription": None,
                "transcription_word_count": 0
            }
        
        # Transcribe the audio
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transcription = transcribe_audio_fast_whisper(
            audio_file_path, 
            model_size="large-v3-turbo", 
            device=device
        )
        
        # Clean up temporary files
        for file_path in cleanup_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {file_path}")
            except OSError as e:
                logger.warning(f"Error removing file {file_path}: {e}")
        
        if transcription:
            logger.info(f"Successfully transcribed record {record_id}")
            
            # Apply text normalization to the transcription
            transcription_result = {
                "extracted_text": transcription
            }
            normalized_result = normalize_result(transcription_result, "extracted_text")
            normalized_transcription = normalized_result["extracted_text"]
            
            # Update metadata with transcription info
            existing_metadata = record.get('metadata')
            if existing_metadata:
                try:
                    metadata_dict = json.loads(existing_metadata) if isinstance(existing_metadata, str) else existing_metadata
                except:
                    metadata_dict = {}
            else:
                metadata_dict = {}
            
            # Add transcription metadata including normalization info
            metadata_dict['transcription_info'] = {
                'transcribed_at': record.get('submission_timestamp'),
                'model_used': 'large-v3-turbo',
                'device_used': device,
                'source_type': content_type,
                'original_word_count': len(transcription.split()),
                'normalized_word_count': len(normalized_transcription.split()),
                'text_normalized': len(transcription) != len(normalized_transcription)
            }
            
            return {
                **record,
                "transcription": normalized_transcription,
                "transcription_status": "success",
                "transcription_error": None,
                "transcription_word_count": len(normalized_transcription.split()),
                "metadata": json.dumps(metadata_dict)
            }
        else:
            return {
                **record,
                "transcription_status": "failed",
                "transcription_error": "Transcription failed",
                "transcription": None,
                "transcription_word_count": 0
            }
            
    except Exception as e:
        logger.error(f"Error processing record {record.get('UUID', record.get('ID', 'unknown'))}: {e}")
        return {
            **record,
            "transcription_status": "error",
            "transcription_error": str(e),
            "transcription": None,
            "transcription_word_count": 0
        }

def process_json_input(json_input: str) -> str:
    """
    Process JSON input containing database record(s).
    
    Args:
        json_input: JSON string containing record or list of records
        
    Returns:
        JSON string with processed results
    """
    try:
        data = json.loads(json_input)
        
        if isinstance(data, list):
            # Process multiple records
            results = []
            for record in data:
                result = process_database_record(record)
                results.append(result)
            return json.dumps(results, indent=2)
        else:
            # Process single record
            result = process_database_record(data)
            return json.dumps(result, indent=2)
            
    except json.JSONDecodeError as e:
        error_result = {
            "error": f"Invalid JSON input: {e}",
            "transcription_status": "error"
        }
        return json.dumps(error_result, indent=2)
    except Exception as e:
        error_result = {
            "error": f"Processing error: {e}",
            "transcription_status": "error"
        }
        return json.dumps(error_result, indent=2)

def main():
    """
    Main function for command-line usage and testing.
    """
    import sys
    
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