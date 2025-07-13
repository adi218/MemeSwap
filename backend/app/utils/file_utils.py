import os
import uuid
from datetime import datetime
from typing import Optional
from fastapi import UploadFile, HTTPException
from app.core.config import settings

def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image. Received: {file.content_type}"
        )
    
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum allowed: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
        )

def generate_unique_filename(prefix: str, extension: str = ".jpg") -> str:
    """Generate a unique filename with timestamp and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}{extension}"

def save_uploaded_file(file_data: bytes, filename: str) -> str:
    """Save uploaded file to uploads directory."""
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(file_data)
    
    return file_path

def cleanup_old_files(directory: str, max_age_hours: int = 24) -> None:
    """Clean up old files from uploads directory."""
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # File might be in use

def get_file_extension(content_type: str) -> str:
    """Get file extension from content type."""
    content_type_to_extension = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg", 
        "image/png": ".png",
        "image/gif": ".gif"
    }
    return content_type_to_extension.get(content_type, ".jpg") 