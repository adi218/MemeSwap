import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

class FaceSwapService:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_face_landmarks_enhanced(self, image: np.ndarray) -> List[Tuple[np.ndarray, List]]:
        """
        Extract face landmarks from an image with enhanced detection including hair and ears.
        Returns list of (face_image, landmarks) tuples.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to numpy array
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    landmarks.append([x, y])
                
                landmarks = np.array(landmarks)
                
                # Get face bounding box with extended area for hair and ears
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                
                # Calculate face dimensions
                face_width = x_max - x_min
                face_height = y_max - y_min
                
                # Extend bounds to include hair and ears
                ear_extension = int(face_width * 0.2)  # 20% extension for ears
                hair_extension = int(face_height * 0.3)  # 30% extension for hair
                chin_extension = int(face_height * 0.1)  # 10% extension for chin
                
                # Apply extensions with boundary checks
                x_min = max(0, x_min - ear_extension)
                x_max = min(image.shape[1], x_max + ear_extension)
                y_min = max(0, y_min - hair_extension)
                y_max = min(image.shape[0], y_max + chin_extension)
                
                # Extract extended face region
                face_image = image[y_min:y_max, x_min:x_max]
                
                # Adjust landmarks to face region
                adjusted_landmarks = landmarks - np.array([x_min, y_min])
                
                faces.append((face_image, adjusted_landmarks))
        
        return faces
    
    def extract_face_landmarks(self, image: np.ndarray) -> List[Tuple[np.ndarray, List]]:
        """
        Extract face landmarks from an image (original method for backward compatibility).
        Returns list of (face_image, landmarks) tuples.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to numpy array
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    landmarks.append([x, y])
                
                landmarks = np.array(landmarks)
                
                # Get face bounding box
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                
                # Extract face region with padding
                padding = 50
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)
                
                face_image = image[y_min:y_max, x_min:x_max]
                
                # Adjust landmarks to face region
                adjusted_landmarks = landmarks - np.array([x_min, y_min])
                
                faces.append((face_image, adjusted_landmarks))
        
        return faces
    
    def detect_faces_in_gif_enhanced(self, gif_path: str) -> List[Tuple[np.ndarray, List, int]]:
        """
        Detect faces in each frame of a GIF with enhanced detection.
        Returns list of (frame, landmarks, frame_index) tuples.
        """
        cap = cv2.VideoCapture(gif_path)
        frames_with_faces = []
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert landmarks to numpy array
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        landmarks.append([x, y])
                    
                    landmarks = np.array(landmarks)
                    frames_with_faces.append((frame, landmarks, frame_index))
            
            frame_index += 1
        
        cap.release()
        return frames_with_faces
    
    def detect_faces_in_gif(self, gif_path: str) -> List[Tuple[np.ndarray, List, int]]:
        """
        Detect faces in each frame of a GIF (original method for backward compatibility).
        Returns list of (frame, landmarks, frame_index) tuples.
        """
        cap = cv2.VideoCapture(gif_path)
        frames_with_faces = []
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert landmarks to numpy array
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        landmarks.append([x, y])
                    
                    landmarks = np.array(landmarks)
                    frames_with_faces.append((frame, landmarks, frame_index))
            
            frame_index += 1
        
        cap.release()
        return frames_with_faces
    
    def warp_face(self, source_face: np.ndarray, source_landmarks: np.ndarray,
                  target_landmarks: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Warp source face to match target face landmarks.
        """
        # Define key facial landmarks for transformation
        # MediaPipe face mesh has 468 landmarks, we'll use key points
        key_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                       397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                       172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        source_key_points = source_landmarks[key_indices]
        target_key_points = target_landmarks[key_indices]
        
        # Calculate transformation matrix
        transformation_matrix = cv2.estimateAffinePartial2D(
            source_key_points, target_key_points
        )[0]
        
        # Warp the source face
        warped_face = cv2.warpAffine(
            source_face, transformation_matrix, (target_shape[1], target_shape[0])
        )
        
        return warped_face
    
    def blend_faces(self, warped_face: np.ndarray, target_frame: np.ndarray,
                   target_landmarks: np.ndarray) -> np.ndarray:
        """
        Blend the warped face into the target frame.
        """
        # Create a mask for the face region
        hull = cv2.convexHull(target_landmarks)
        mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        
        # Apply Gaussian blur to the mask for smooth blending
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Normalize mask
        mask = mask.astype(np.float32) / 255.0
        mask = np.stack([mask] * 3, axis=2)
        
        # Blend the faces
        blended_frame = (warped_face * mask + target_frame * (1 - mask)).astype(np.uint8)
        
        return blended_frame
    
    def swap_face_on_gif(self, source_image_path: str, target_gif_path: str, enhanced: bool = True) -> str:
        """
        Swap faces from source image onto target GIF with enhanced detection.
        Returns path to the output GIF.
        """
        try:
            # Load source image
            source_image = cv2.imread(source_image_path)
            if source_image is None:
                raise ValueError("Could not load source image")
            
            # Extract faces from source image using enhanced or standard detection
            if enhanced:
                source_faces = self.extract_face_landmarks_enhanced(source_image)
            else:
                source_faces = self.extract_face_landmarks(source_image)
                
            if not source_faces:
                raise ValueError("No faces detected in source image")
            
            # Use the first detected face as the source
            source_face, source_landmarks = source_faces[0]
            
            # Detect faces in target GIF using enhanced or standard detection
            if enhanced:
                frames_with_faces = self.detect_faces_in_gif_enhanced(target_gif_path)
            else:
                frames_with_faces = self.detect_faces_in_gif(target_gif_path)
                
            if not frames_with_faces:
                raise ValueError("No faces detected in target GIF")
            
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
                    for target_frame, target_landmarks, _ in frame_faces:
                        # Warp source face to match target
                        warped_face = self.warp_face(
                            source_face, source_landmarks, target_landmarks, frame.shape[:2]
                        )
                        
                        # Blend the faces
                        frame = self.blend_faces(warped_face, frame, target_landmarks)
                
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
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in face swap: {str(e)}")
            raise
    
    def swap_face_on_image(self, source_image_path: str, target_image_path: str, enhanced: bool = True) -> str:
        """
        Swap faces from source image onto target image with enhanced detection.
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
            
            # Extract faces from source image using enhanced or standard detection
            if enhanced:
                source_faces = self.extract_face_landmarks_enhanced(source_image)
            else:
                source_faces = self.extract_face_landmarks(source_image)
                
            if not source_faces:
                raise ValueError("No faces detected in source image")
            
            # Use the first detected face as the source
            source_face, source_landmarks = source_faces[0]
            
            # Extract faces from target image using enhanced or standard detection
            if enhanced:
                target_faces = self.extract_face_landmarks_enhanced(target_image)
            else:
                target_faces = self.extract_face_landmarks(target_image)
                
            if not target_faces:
                raise ValueError("No faces detected in target image")
            
            # Process each face in target image
            result_image = target_image.copy()
            
            for target_face, target_landmarks in target_faces:
                # Warp source face to match target
                warped_face = self.warp_face(
                    source_face, source_landmarks, target_landmarks, target_image.shape[:2]
                )
                
                # Blend the faces
                result_image = self.blend_faces(warped_face, result_image, target_landmarks)
            
            # Save result
            output_path = target_image_path.replace('.', '_swapped.')
            cv2.imwrite(output_path, result_image)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in face swap: {str(e)}")
            raise 