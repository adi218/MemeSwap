import os
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # API Configuration
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "MemeSwap API"
    VERSION: str = "1.0.0"
    
    # CORS Configuration
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ]
    
    # Tenor API Configuration
    TENOR_API_KEY: Optional[str] = None
    
    # File Upload Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/jpg", 
        "image/png",
        "image/gif"
    ]
    

    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from environment variables

# Create settings instance
settings = Settings()

# Parse ALLOWED_ORIGINS string into list if provided
if settings.ALLOWED_ORIGINS:
    settings.BACKEND_CORS_ORIGINS = [
        origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")
    ]

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True) 