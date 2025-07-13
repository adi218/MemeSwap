from fastapi import APIRouter, HTTPException, UploadFile, File, Query
import requests
import base64
from typing import Optional, Literal
from app.ml.face_swap import FaceSwapService
from app.ml.face_detection import FaceDetectionService
from app.utils.file_utils import (
    validate_image_file, 
    generate_unique_filename, 
    save_uploaded_file,
    get_file_extension
)
from app.core.config import settings

router = APIRouter()
face_swap_service = FaceSwapService()
face_detection_service = FaceDetectionService()

@router.post("/swap-face-on-gif")
async def swap_face_on_gif(
    source_image: UploadFile = File(...),
    gif_url: str = Query(..., description="URL of the target GIF"),
    model: Literal["mediapipe_standard", "mediapipe_enhanced", "yolo"] = Query("mediapipe_enhanced", description="Face detection model to use")
):
    """
    Swap a face from source image onto a GIF.
    
    Args:
        source_image: The source image containing the face to swap
        gif_url: URL of the target GIF to swap faces onto
        model: Face detection model to use
    
    Returns:
        JSON response with the face-swapped GIF data
    """
    # Validate source file
    validate_image_file(source_image)
    
    try:
        # Read source image
        source_image_data = await source_image.read()
        
        # Generate unique filename for source image
        extension = get_file_extension(source_image.content_type)
        source_filename = generate_unique_filename("source", extension)
        source_path = save_uploaded_file(source_image_data, source_filename)
        
        # Download GIF
        gif_response = requests.get(gif_url, timeout=30)
        gif_response.raise_for_status()
        
        gif_filename = generate_unique_filename("target", ".gif")
        gif_path = save_uploaded_file(gif_response.content, gif_filename)
        
        # Perform face swap with selected model
        enhanced = model == "mediapipe_enhanced"
        output_path = face_swap_service.swap_face_on_gif(source_path, gif_path, enhanced=enhanced)
        
        # Read the output file and convert to base64
        with open(output_path, "rb") as f:
            output_data = f.read()
        
        output_base64 = base64.b64encode(output_data).decode('utf-8')
        
        return {
            "message": "Face swap completed successfully",
            "output_path": output_path,
            "output_data": output_base64,
            "source_image": source_filename,
            "target_gif": gif_filename,
            "model_used": model
        }
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download GIF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing face swap: {str(e)}")

@router.post("/swap-face-on-image")
async def swap_face_on_image(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    model: Literal["mediapipe_standard", "mediapipe_enhanced", "yolo"] = Query("mediapipe_enhanced", description="Face detection model to use")
):
    """
    Swap a face from source image onto target image.
    
    Args:
        source_image: The source image containing the face to swap
        target_image: The target image to swap faces onto
        model: Face detection model to use
    
    Returns:
        JSON response with the face-swapped image data
    """
    # Validate files
    validate_image_file(source_image)
    validate_image_file(target_image)
    
    try:
        # Read images
        source_image_data = await source_image.read()
        target_image_data = await target_image.read()
        
        # Generate unique filenames
        source_extension = get_file_extension(source_image.content_type)
        target_extension = get_file_extension(target_image.content_type)
        source_filename = generate_unique_filename("source", source_extension)
        target_filename = generate_unique_filename("target", target_extension)
        
        source_path = save_uploaded_file(source_image_data, source_filename)
        target_path = save_uploaded_file(target_image_data, target_filename)
        
        # Perform face swap with selected model
        enhanced = model == "mediapipe_enhanced"
        output_path = face_swap_service.swap_face_on_image(source_path, target_path, enhanced=enhanced)
        
        # Read the output file and convert to base64
        with open(output_path, "rb") as f:
            output_data = f.read()
        
        output_base64 = base64.b64encode(output_data).decode('utf-8')
        
        return {
            "message": "Face swap completed successfully",
            "output_path": output_path,
            "output_data": output_base64,
            "source_image": source_filename,
            "target_image": target_filename,
            "model_used": model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing face swap: {str(e)}")

@router.post("/detect-faces")
async def detect_faces(
    file: UploadFile = File(...),
    model: Literal["mediapipe_standard", "mediapipe_enhanced", "yolo"] = Query("mediapipe_enhanced", description="Face detection model to use")
):
    """
    Detect faces in an uploaded image with model selection.
    
    Args:
        file: The image file to detect faces in
        model: Face detection model to use
    
    Returns:
        JSON response with face detection results
    """
    validate_image_file(file)
    
    try:
        # Read file content
        image_data = await file.read()
        
        # Detect faces using selected model
        detection_result = face_detection_service.detect_faces_with_model(image_data, model)
        
        # Get annotated image
        annotated_result = face_detection_service.draw_faces_on_image(image_data, model=model)
        
        return {
            "faces_found": detection_result["faces_found"],
            "faces": detection_result["faces"],
            "annotated_image": annotated_result["annotated_image"],
            "image_shape": detection_result.get("image_shape", {}),
            "model_used": model,
            "message": f"Detected {detection_result['faces_found']} faces using {model} model"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting faces: {str(e)}")

@router.post("/upload-face")
async def upload_face(
    file: UploadFile = File(...),
    model: Literal["mediapipe_standard", "mediapipe_enhanced", "yolo"] = Query("mediapipe_enhanced", description="Face detection model to use")
):
    """
    Upload an image and detect faces in it with model selection.
    
    Args:
        file: The image file to upload and process
        model: Face detection model to use
    
    Returns:
        JSON response with upload and detection results
    """
    validate_image_file(file)
    
    try:
        # Read file content
        image_data = await file.read()
        
        # Generate unique filename
        extension = get_file_extension(file.content_type)
        filename = generate_unique_filename("face", extension)
        upload_path = save_uploaded_file(image_data, filename)
        
        # Detect faces using selected model
        detection_result = face_detection_service.detect_faces_with_model(image_data, model)
        
        # Get annotated image
        annotated_result = face_detection_service.draw_faces_on_image(image_data, model=model)
        
        return {
            "filename": filename,
            "upload_path": upload_path,
            "faces_found": detection_result["faces_found"],
            "faces": detection_result["faces"],
            "annotated_image": annotated_result["annotated_image"],
            "image_shape": detection_result.get("image_shape", {}),
            "model_used": model,
            "message": f"Successfully uploaded and detected {detection_result['faces_found']} faces using {model} model"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/available-models")
async def get_available_models():
    """
    Get list of available face detection models and their capabilities.
    
    Returns:
        JSON response with available models and their features
    """
    models = {
        "mediapipe_standard": {
            "name": "MediaPipe Standard",
            "description": "Basic MediaPipe face detection with bounding boxes",
            "features": ["Fast detection", "Basic bounding boxes", "Key points"],
            "best_for": "Quick detection, basic face detection"
        },
        "mediapipe_enhanced": {
            "name": "MediaPipe Enhanced",
            "description": "Advanced MediaPipe face mesh with 468 landmarks including hair and ears",
            "features": ["468 facial landmarks", "Includes hair and ears", "Precise detection", "Best for face swapping"],
            "best_for": "High-quality face swapping, precise detection"
        },
        "yolo": {
            "name": "YOLO Face Detection",
            "description": "YOLO-based face detection for robust detection",
            "features": ["Robust detection", "Good for various angles", "Fast inference"],
            "best_for": "Robust detection, various face angles",
            "available": face_detection_service.yolo_available
        }
    }
    
    return {
        "models": models,
        "default_model": "mediapipe_enhanced"
    } 