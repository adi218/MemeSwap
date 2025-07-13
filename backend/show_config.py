#!/usr/bin/env python3
"""
Script to display all configurable values from the settings.
Run this to see all available configuration options.
"""

from app.core.config import settings
from app.ml.config import ml_settings

def show_config():
    """Display all configuration values"""
    print("🔧 MemeSwap Configuration")
    print("=" * 50)
    
    print("\n📡 Server Configuration:")
    print(f"  HOST: {settings.HOST}")
    print(f"  PORT: {settings.PORT}")
    
    print("\n🌐 CORS Configuration:")
    print(f"  ALLOWED_ORIGINS: {settings.ALLOWED_ORIGINS}")
    print(f"  BACKEND_CORS_ORIGINS: {settings.BACKEND_CORS_ORIGINS}")
    
    print("\n📁 File Upload Configuration:")
    print(f"  UPLOAD_DIR: {settings.UPLOAD_DIR}")
    print(f"  MAX_FILE_SIZE: {settings.MAX_FILE_SIZE} bytes")
    print(f"  ALLOWED_IMAGE_TYPES: {settings.ALLOWED_IMAGE_TYPES}")
    
    print("\n👤 Face Detection Configuration:")
    print(f"  FACE_DETECTION_CONFIDENCE: {ml_settings.FACE_DETECTION_CONFIDENCE}")
    print(f"  MAX_FACES_PER_IMAGE: {ml_settings.MAX_FACES_PER_IMAGE}")
    
    print("\n🎯 MediaPipe Face Detection Configuration:")
    print(f"  MEDIAPIPE_CONFIDENCE_THRESHOLD: {ml_settings.MEDIAPIPE_CONFIDENCE_THRESHOLD}")
    print(f"  MEDIAPIPE_DROP_THRESHOLD: {ml_settings.MEDIAPIPE_DROP_THRESHOLD}")
    print(f"  MEDIAPIPE_ENHANCED_CONFIDENCE: {ml_settings.MEDIAPIPE_ENHANCED_CONFIDENCE}")
    print(f"  MEDIAPIPE_STANDARD_CONFIDENCE: {ml_settings.MEDIAPIPE_STANDARD_CONFIDENCE}")
    print(f"  MEDIAPIPE_FACE_MESH_CONFIDENCE: {ml_settings.MEDIAPIPE_FACE_MESH_CONFIDENCE}")
    print(f"  MEDIAPIPE_DETECTION_CONFIDENCE: {ml_settings.MEDIAPIPE_DETECTION_CONFIDENCE}")
    
    print("\n🤖 YOLO Face Detection Configuration:")
    print(f"  YOLO_CONFIDENCE_THRESHOLD: {ml_settings.YOLO_CONFIDENCE_THRESHOLD}")
    print(f"  YOLO_NMS_THRESHOLD: {ml_settings.YOLO_NMS_THRESHOLD}")
    print(f"  YOLO_MODEL_PATH: {ml_settings.YOLO_MODEL_PATH}")
    
    print("\n🔄 Face Swap Configuration:")
    print(f"  FACE_SWAP_QUALITY: {ml_settings.FACE_SWAP_QUALITY}")
    print(f"  FACE_SWAP_DURATION: {ml_settings.FACE_SWAP_DURATION} ms")
    print(f"  FACE_SWAP_DEFAULT_CONFIDENCE: {ml_settings.FACE_SWAP_DEFAULT_CONFIDENCE}")
    print(f"  FACE_SWAP_DEFAULT_DROP_THRESHOLD: {ml_settings.FACE_SWAP_DEFAULT_DROP_THRESHOLD}")
    print(f"  FACE_SWAP_DEBUG_SAVE_FACES: {ml_settings.FACE_SWAP_DEBUG_SAVE_FACES}")
    print(f"  FACE_SWAP_DEBUG_MAX_FACES: {ml_settings.FACE_SWAP_DEBUG_MAX_FACES}")
    
    print("\n📐 Pose Estimation Configuration:")
    print(f"  POSE_ESTIMATION_CONFIDENCE: {ml_settings.POSE_ESTIMATION_CONFIDENCE}")
    print(f"  POSE_ESTIMATION_DISTANCE_THRESHOLD: {ml_settings.POSE_ESTIMATION_DISTANCE_THRESHOLD}")
    
    print("\n🔑 API Configuration:")
    print(f"  TENOR_API_KEY: {'Set' if settings.TENOR_API_KEY else 'Not set'}")
    
    print("\n" + "=" * 50)
    print("💡 To override these values, set environment variables or modify .env file")
    print("   Example: MEDIAPIPE_CONFIDENCE_THRESHOLD=0.4")

if __name__ == "__main__":
    show_config() 