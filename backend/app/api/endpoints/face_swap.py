from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Literal
import base64
import requests
import logging
from app.utils.file_utils import validate_image_file, get_file_extension, generate_unique_filename, save_uploaded_file
from app.ml.face_swap import FaceSwapService
from app.ml.face_detection import FaceDetectionService

logger = logging.getLogger(__name__)

# Initialize services
face_detection_service = FaceDetectionService()
face_swap_service = FaceSwapService()
face_swap_service.set_face_detection_service(face_detection_service)

router = APIRouter()

@router.post("/swap-face-on-gif")
async def swap_face_on_gif(
    source_image: UploadFile = File(...),
    gif_url: str = Query(..., description="URL of the target GIF")
):
    """
    Swap a face from source image onto a GIF using YOLO detection.
    
    Args:
        source_image: The source image containing the face to swap
        gif_url: URL of the target GIF to swap faces onto
    
    Returns:
        JSON response with the face-swapped GIF data
    """
    logger.info(f"=== FACE SWAP REQUEST ===")
    logger.info(f"GIF URL: {gif_url}")
    logger.info(f"Source image: {source_image.filename}, size: {source_image.size}")
    
    # Validate source file
    validate_image_file(source_image)
    
    try:
        # Read source image
        source_image_data = await source_image.read()
        logger.info(f"Source image data size: {len(source_image_data)} bytes")
        
        # Generate unique filename for source image
        extension = get_file_extension(source_image.content_type)
        source_filename = generate_unique_filename("source", extension)
        source_path = save_uploaded_file(source_image_data, source_filename)
        logger.info(f"Saved source image to: {source_path}")
        
        # Download GIF
        logger.info(f"Downloading GIF from: {gif_url}")
        gif_response = requests.get(gif_url, timeout=30)
        gif_response.raise_for_status()
        logger.info(f"GIF downloaded successfully, size: {len(gif_response.content)} bytes")
        
        gif_filename = generate_unique_filename("target", ".gif")
        gif_path = save_uploaded_file(gif_response.content, gif_filename)
        logger.info(f"Saved GIF to: {gif_path}")
        
        # Perform face swap using YOLO detection
        logger.info("Starting face swap with YOLO detection")
        
        output_path = face_swap_service.swap_face_on_gif(source_path, gif_path)
        logger.info(f"Face swap completed, output saved to: {output_path}")
        
        # Read the output file and convert to base64
        with open(output_path, "rb") as f:
            output_data = f.read()
        
        output_base64 = base64.b64encode(output_data).decode('utf-8')
        logger.info(f"Output converted to base64, size: {len(output_base64)} characters")
        
        logger.info("=== FACE SWAP COMPLETED SUCCESSFULLY ===")
        
        return {
            "message": "Face swap completed successfully",
            "output_path": output_path,
            "output_data": output_base64,
            "source_image": source_filename,
            "target_gif": gif_filename,
            "model_used": "yolo"
        }
        
    except requests.RequestException as e:
        logger.error(f"Failed to download GIF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download GIF: {str(e)}")
    except Exception as e:
        logger.error(f"Error performing face swap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing face swap: {str(e)}")

@router.post("/swap-face-on-image")
async def swap_face_on_image(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...)
):
    """
    Swap a face from source image onto target image using YOLO detection.
    
    Args:
        source_image: The source image containing the face to swap
        target_image: The target image to swap faces onto
    
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
        
        # Perform face swap using YOLO detection
        output_path = face_swap_service.swap_face_on_image(source_path, target_path)
        
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
            "model_used": "yolo"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing face swap: {str(e)}")

@router.post("/detect-faces")
async def detect_faces(
    file: UploadFile = File(...),
    model: Literal["yolo"] = Query("yolo", description="Face detection model to use (YOLO only)")
):
    """
    Detect faces in an uploaded image using YOLO detection.
    
    Args:
        file: The image file to detect faces in
        model: Face detection model to use (YOLO only)
    
    Returns:
        JSON response with face detection results
    """
    logger.info(f"=== FACE DETECTION REQUEST ===")
    logger.info(f"Model selected: {model}")
    logger.info(f"File: {file.filename}, size: {file.size}, content_type: {file.content_type}")
    
    validate_image_file(file)
    
    try:
        # Read file content
        image_data = await file.read()
        logger.info(f"Image data size: {len(image_data)} bytes")
        
        # Detect faces using YOLO model
        logger.info(f"Calling face detection with model: {model}")
        detection_result = face_detection_service.detect_faces_with_model(image_data, model)
        logger.info(f"Detection result: {detection_result}")
        
        # Get annotated image
        logger.info("Generating annotated image...")
        annotated_result = face_detection_service.draw_faces_on_image(image_data, model=model)
        logger.info(f"Annotated image generated, faces found: {annotated_result.get('faces_found', 0)}")
        
        logger.info("=== FACE DETECTION COMPLETED SUCCESSFULLY ===")
        
        return {
            "faces_found": detection_result["faces_found"],
            "faces": detection_result["faces"],
            "annotated_image": annotated_result["annotated_image"],
            "image_shape": detection_result.get("image_shape", {}),
            "model_used": model,
            "message": f"Detected {detection_result['faces_found']} faces using {model} model"
        }
        
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting faces: {str(e)}")

@router.post("/upload-face")
async def upload_face(
    file: UploadFile = File(...),
    model: Literal["yolo"] = Query("yolo", description="Face detection model to use (YOLO only)")
):
    """
    Upload an image and detect faces in it using YOLO detection.
    
    Args:
        file: The image file to upload and process
        model: Face detection model to use (YOLO only)
    
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
        
        # Detect faces using YOLO model
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

@router.post("/estimate-pose")
async def estimate_face_pose(
    file: UploadFile = File(...),
    model: Literal["yolo"] = Query("yolo", description="Face detection model to use (YOLO only)")
):
    """
    Estimate face pose (yaw, pitch, roll) from an uploaded image using YOLO detection.
    
    Args:
        file: The image file to analyze
        model: Face detection model to use (YOLO only)
    
    Returns:
        JSON response with pose estimation results
    """
    validate_image_file(file)
    
    try:
        # Read file content
        image_data = await file.read()
        
        # Detect faces and get pose information using YOLO
        detection_result = face_detection_service.detect_faces_with_model(image_data, model)
        
        # Extract pose information from detected faces
        pose_results = []
        for face in detection_result.get("faces", []):
            pose_info = face.get("pose", {})
            pose_results.append({
                "face_index": len(pose_results),
                "bbox": face.get("bbox", {}),
                "confidence": face.get("confidence", 0.0),
                "pose": {
                    "yaw": pose_info.get("yaw", 0.0),
                    "pitch": pose_info.get("pitch", 0.0),
                    "roll": pose_info.get("roll", 0.0),
                    "pose_confidence": pose_info.get("confidence", 0.0),
                    "success": pose_info.get("success", False)
                }
            })
        
        return {
            "faces_found": detection_result["faces_found"],
            "pose_results": pose_results,
            "model_used": model,
            "message": f"Estimated pose for {len(pose_results)} faces using {model} model"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error estimating pose: {str(e)}")

@router.get("/available-models")
async def get_available_models():
    """
    Get list of available face detection models and their capabilities.
    
    Returns:
        JSON response with available models and their features
    """
    logger.info("=== AVAILABLE MODELS REQUEST ===")
    logger.info(f"YOLO available: {face_detection_service.yolo_available}")
    
    models = {
        "yolo": {
            "name": "YOLO Face Detection",
            "description": "YOLO-based face detection with pose estimation for robust detection",
            "features": ["Robust detection", "Good for various angles", "Fast inference", "Pose estimation", "Best for face swapping"],
            "best_for": "Robust detection, various face angles with pose analysis, face swapping",
            "available": face_detection_service.yolo_available
        }
    }
    
    logger.info("=== AVAILABLE MODELS RESPONSE ===")
    return {
        "models": models,
        "default_model": "yolo"
    } 