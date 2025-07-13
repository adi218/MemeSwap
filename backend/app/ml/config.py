import os
from typing import Optional
from pydantic_settings import BaseSettings

class MLSettings(BaseSettings):
    """ML-specific settings and configuration."""
    
    # Face Detection Configuration
    FACE_DETECTION_CONFIDENCE: float = 0.5
    MAX_FACES_PER_IMAGE: int = 10
    
    # MediaPipe Face Detection Configuration
    MEDIAPIPE_CONFIDENCE_THRESHOLD: float = 0.3
    MEDIAPIPE_DROP_THRESHOLD: int = 3
    MEDIAPIPE_ENHANCED_CONFIDENCE: float = 0.3
    MEDIAPIPE_STANDARD_CONFIDENCE: float = 0.5
    MEDIAPIPE_FACE_MESH_CONFIDENCE: float = 0.5
    MEDIAPIPE_DETECTION_CONFIDENCE: float = 0.5
    
    # YOLO Face Detection Configuration
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_NMS_THRESHOLD: float = 0.4
    YOLO_MODEL_PATH: str = "models/yolov8n-face.pt"
    
    # Face Swap Configuration
    FACE_SWAP_QUALITY: int = 85  # GIF quality
    FACE_SWAP_DURATION: int = 100  # Frame duration in ms
    FACE_SWAP_DEFAULT_CONFIDENCE: float = 0.3
    FACE_SWAP_DEFAULT_DROP_THRESHOLD: int = 3
    FACE_SWAP_DEBUG_SAVE_FACES: bool = True
    FACE_SWAP_DEBUG_MAX_FACES: int = 4
    
    # Pose Estimation Configuration
    POSE_ESTIMATION_CONFIDENCE: float = 0.5
    POSE_ESTIMATION_DISTANCE_THRESHOLD: float = 0.1
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from environment variables

# Create ML settings instance
ml_settings = MLSettings()

# Ensure model directory exists
os.makedirs(os.path.dirname(ml_settings.YOLO_MODEL_PATH), exist_ok=True) 