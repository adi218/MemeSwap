import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Tuple, Optional
import logging
import os
import uuid

logger = logging.getLogger(__name__)

class FaceSwapService:
    def __init__(self):
        # Initialize YOLO model for face detection
        self.yolo_model = None
        self.face_detection_service = None
        
    def set_face_detection_service(self, face_detection_service):
        """Set the face detection service to use for YOLO detection"""
        self.face_detection_service = face_detection_service
    
    def extract_face_with_mediapipe(self, image: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
        """
        Extract faces from an image using MediaPipe enhanced detection.
        Returns list of (face_image, face_info) tuples.
        """
        if not self.face_detection_service:
            raise ValueError("Face detection service not set")
        
        # Convert image to bytes for the detection service
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode image")
        
        image_bytes = buffer.tobytes()
        
        # Try MediaPipe enhanced detection first
        detection_result = self.face_detection_service.detect_faces_with_model(image_bytes, "mediapipe_enhanced")
        
        faces = []
        if detection_result["faces_found"] > 0:
            for face in detection_result["faces"]:
                bbox = face["bbox"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                
                # Extract face region with some padding
                padding = int(min(w, h) * 0.1)  # 10% padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                face_image = image[y1:y2, x1:x2]
                
                # Create face info dictionary
                face_info = {
                    "bbox": bbox,
                    "confidence": face.get("confidence", 0.0),
                    "pose": face.get("pose", {}),
                    "model": "mediapipe_enhanced"
                }
                
                faces.append((face_image, face_info))
        else:
            # If MediaPipe enhanced fails, try standard MediaPipe detection
            logger.info("MediaPipe enhanced detection failed, trying standard MediaPipe detection...")
            detection_result = self.face_detection_service.detect_faces_with_model(image_bytes, "mediapipe_standard")
            
            if detection_result["faces_found"] > 0:
                for face in detection_result["faces"]:
                    bbox = face["bbox"]
                    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                    
                    # Extract face region with some padding
                    padding = int(min(w, h) * 0.1)  # 10% padding
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)
                    
                    face_image = image[y1:y2, x1:x2]
                    
                    # Create face info dictionary
                    face_info = {
                        "bbox": bbox,
                        "confidence": face.get("confidence", 0.0),
                        "pose": face.get("pose", {}),
                        "model": "mediapipe_standard"
                    }
                    
                    faces.append((face_image, face_info))
        
        return faces
    
    def detect_faces_in_gif_with_mediapipe(self, gif_path: str) -> List[Tuple[np.ndarray, dict, int]]:
        """
        Detect faces in each frame of a GIF using MediaPipe enhanced detection.
        Returns list of (frame, face_info, frame_index) tuples.
        """
        if not self.face_detection_service:
            raise ValueError("Face detection service not set")
        
        cap = cv2.VideoCapture(gif_path)
        frames_with_faces = []
        frame_index = 0
        last_face_info = None  # Store last detected face for interpolation
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use MediaPipe enhanced detection on this frame
            faces = self.extract_face_with_mediapipe(frame)
            
            if faces:
                # Face detected in this frame
                for face_image, face_info in faces:
                    frames_with_faces.append((frame, face_info, frame_index))
                    last_face_info = face_info  # Update last known face
            elif last_face_info is not None:
                # No face detected, but we have a previous face - use interpolation
                # Slightly adjust the bounding box position based on frame movement
                interpolated_face_info = last_face_info.copy()
                frames_with_faces.append((frame, interpolated_face_info, frame_index))
                logger.info(f"Frame {frame_index}: No face detected, using interpolated face from previous frame")
            
            frame_index += 1
        
        cap.release()
        return frames_with_faces
    
    def save_cropped_faces(self, faces: List[Tuple[np.ndarray, dict]], prefix: str, output_dir: str = "debug_faces") -> List[str]:
        """
        Save cropped faces to disk for debugging.
        Returns list of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, (face_image, face_info) in enumerate(faces):
            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{prefix}_face_{i}_{unique_id}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save the face image
            cv2.imwrite(filepath, face_image)
            saved_paths.append(filepath)
            
            logger.info(f"Saved {prefix} face {i} to {filepath}")
            logger.info(f"Face info: bbox={face_info['bbox']}, confidence={face_info['confidence']:.3f}")
        
        return saved_paths
    
    def warp_face(self, source_image: np.ndarray, source_face_info: dict,
                  target_face_info: dict, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Warp source face to match target face using bounding boxes.
        """
        # Get bounding box coordinates
        source_bbox = source_face_info["bbox"]
        target_bbox = target_face_info["bbox"]
        
        sx, sy, sw, sh = source_bbox["x"], source_bbox["y"], source_bbox["width"], source_bbox["height"]
        tx, ty, tw, th = target_bbox["x"], target_bbox["y"], target_bbox["width"], target_bbox["height"]
        
        # Extract source face region from the full source image
        sx1 = max(0, sx)
        sy1 = max(0, sy)
        sx2 = min(source_image.shape[1], sx + sw)
        sy2 = min(source_image.shape[0], sy + sh)
        
        source_face_region = source_image[sy1:sy2, sx1:sx2]
        
        # Resize source face region to match target size
        resized_face = cv2.resize(source_face_region, (tw, th))
        
        # Create output image with proper shape (height, width, channels)
        if len(resized_face.shape) == 3:
            warped_face = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)
        else:
            warped_face = np.zeros(target_shape, dtype=np.uint8)
        
        # Place the resized face at the target location
        y1 = max(0, ty)
        y2 = min(target_shape[0], ty + th)
        x1 = max(0, tx)
        x2 = min(target_shape[1], tx + tw)
        
        if y2 > y1 and x2 > x1:
            # Ensure the regions have matching shapes
            face_region = resized_face[:y2-y1, :x2-x1]
            if len(face_region.shape) == 3 and len(warped_face.shape) == 3:
                warped_face[y1:y2, x1:x2] = face_region
            elif len(face_region.shape) == 2 and len(warped_face.shape) == 2:
                warped_face[y1:y2, x1:x2] = face_region
            else:
                # Convert grayscale to RGB if needed
                if len(face_region.shape) == 2 and len(warped_face.shape) == 3:
                    face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_GRAY2RGB)
                    warped_face[y1:y2, x1:x2] = face_region_rgb
                else:
                    # Convert RGB to grayscale if needed
                    face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
                    warped_face[y1:y2, x1:x2] = face_region_gray
        
        return warped_face
    
    def blend_faces(self, warped_face: np.ndarray, target_frame: np.ndarray,
                   target_bbox: dict) -> np.ndarray:
        """
        Blend the warped face into the target frame.
        """
        # Create a mask for the face region
        tx, ty, tw, th = target_bbox["x"], target_bbox["y"], target_bbox["width"], target_bbox["height"]
        
        # Create elliptical mask for smooth blending
        mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
        center = (tx + tw // 2, ty + th // 2)
        axes = (tw // 2, th // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Apply Gaussian blur to the mask for smooth blending
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Normalize mask and ensure it has the same shape as the images
        mask = mask.astype(np.float32) / 255.0
        
        # Ensure warped_face and target_frame have the same shape
        if warped_face.shape != target_frame.shape:
            # Resize warped_face to match target_frame shape
            warped_face = cv2.resize(warped_face, (target_frame.shape[1], target_frame.shape[0]))
        
        # Create 3-channel mask for RGB blending
        if len(target_frame.shape) == 3:
            mask_3d = np.stack([mask] * 3, axis=2)
        else:
            mask_3d = mask
        
        # Blend the faces
        blended_frame = (warped_face * mask_3d + target_frame * (1 - mask_3d)).astype(np.uint8)
        
        return blended_frame
    
    def swap_face_on_gif(self, source_image_path: str, target_gif_path: str) -> str:
        """
        Swap faces from source image onto target GIF using YOLO detection.
        Returns path to the output GIF.
        """
        try:
            # Load source image
            source_image = cv2.imread(source_image_path)
            if source_image is None:
                raise ValueError("Could not load source image")
            
            # Extract faces from source image using MediaPipe enhanced detection
            source_faces = self.extract_face_with_mediapipe(source_image)
            if not source_faces:
                raise ValueError("No faces detected in source image")
            
            # Save source faces for debugging
            source_face_paths = self.save_cropped_faces(source_faces, "source")
            logger.info(f"Saved {len(source_face_paths)} source faces")
            
            # Use the first detected face as the source
            source_face, source_face_info = source_faces[0]
            
            # Detect faces in target GIF using MediaPipe enhanced detection
            frames_with_faces = self.detect_faces_in_gif_with_mediapipe(target_gif_path)
            if not frames_with_faces:
                raise ValueError("No faces detected in target GIF")
            
            # Save sample faces from GIF for debugging (up to 4)
            sample_faces = []
            for frame, face_info, frame_idx in frames_with_faces[:4]:
                # Extract the face region from the frame
                bbox = face_info["bbox"]
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                padding = int(min(w, h) * 0.1)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                face_image = frame[y1:y2, x1:x2]
                sample_faces.append((face_image, face_info))
            
            gif_face_paths = self.save_cropped_faces(sample_faces, "gif")
            logger.info(f"Saved {len(gif_face_paths)} sample faces from GIF")
            
            # Process each frame with faces
            cap = cv2.VideoCapture(target_gif_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Create output GIF path
            output_path = target_gif_path.replace('.gif', '_swapped.gif')
            
            # Process frames and collect them
            cap = cv2.VideoCapture(target_gif_path)
            frame_index = 0
            processed_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if this frame has faces to swap
                frame_faces = [f for f in frames_with_faces if f[2] == frame_index]
                
                if frame_faces:
                    # Swap faces in this frame
                    for target_frame, target_face_info, _ in frame_faces:
                        # Warp source face to match target
                        warped_face = self.warp_face(
                            source_image, source_face_info, target_face_info, frame.shape[:2]
                        )
                        
                        # Blend the faces
                        frame = self.blend_faces(warped_face, frame, target_face_info["bbox"])
                
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                processed_frames.append(pil_frame)
                frame_index += 1
            
            cap.release()
            
            # Save as GIF
            if processed_frames:
                # Calculate duration for each frame (in milliseconds)
                duration = int(1000 / fps) if fps > 0 else 100
                
                processed_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=processed_frames[1:],
                    duration=duration,
                    loop=0,
                    optimize=True
                )
            
            logger.info(f"Face swap completed successfully. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in face swap: {str(e)}")
            raise
    
    def swap_face_on_image(self, source_image_path: str, target_image_path: str) -> str:
        """
        Swap faces from source image onto target image using YOLO detection.
        Returns path to the output image.
        """
        try:
            # Load images
            source_image = cv2.imread(source_image_path)
            target_image = cv2.imread(target_image_path)
            
            if source_image is None:
                raise ValueError("Could not load source image")
            if target_image is None:
                raise ValueError("Could not load target image")
            
            # Extract faces from source image using MediaPipe enhanced detection
            source_faces = self.extract_face_with_mediapipe(source_image)
            if not source_faces:
                raise ValueError("No faces detected in source image")
            
            # Save source faces for debugging
            source_face_paths = self.save_cropped_faces(source_faces, "source")
            logger.info(f"Saved {len(source_face_paths)} source faces")
            
            # Use the first detected face as the source
            source_face, source_face_info = source_faces[0]
            
            # Extract faces from target image using MediaPipe enhanced detection
            target_faces = self.extract_face_with_mediapipe(target_image)
            if not target_faces:
                raise ValueError("No faces detected in target image")
            
            # Save target faces for debugging
            target_face_paths = self.save_cropped_faces(target_faces, "target")
            logger.info(f"Saved {len(target_face_paths)} target faces")
            
            # Process each face in target image
            result_image = target_image.copy()
            
            for target_face, target_face_info in target_faces:
                # Warp source face to match target
                warped_face = self.warp_face(
                    source_image, source_face_info, target_face_info, target_image.shape[:2]
                )
                
                # Blend the faces
                result_image = self.blend_faces(warped_face, result_image, target_face_info["bbox"])
            
            # Save result
            output_path = target_image_path.replace('.', '_swapped.')
            cv2.imwrite(output_path, result_image)
            
            logger.info(f"Face swap completed successfully. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in face swap: {str(e)}")
            raise 