import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
import base64
from typing import Literal

class FaceDetectionService:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe face detection and mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # Enhanced face mesh for precise detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize YOLO face detection
        self.yolo_net = None
        self._load_yolo_model()

    def _load_yolo_model(self):
        """Load YOLO face detection model"""
        try:
            # YOLO face detection model path
            model_path = "models/yolov8n-face.pt"
            
            # Try to load YOLO model if available
            try:
                import torch
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)
                self.yolo_available = True
            except ImportError:
                print("Ultralytics not available, YOLO detection disabled")
                self.yolo_available = False
            except Exception as e:
                print(f"Could not load YOLO model: {e}")
                self.yolo_available = False
                
        except Exception as e:
            print(f"YOLO initialization error: {e}")
            self.yolo_available = False

    def detect_faces_yolo(self, image_data):
        """
        Detect faces using YOLO model
        """
        if not self.yolo_available:
            return {
                "error": "YOLO model not available",
                "faces_found": 0,
                "faces": []
            }
        
        try:
            # Convert image data to numpy array
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

            # Run YOLO detection
            results = self.yolo_model(image)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Convert to integer coordinates
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        faces.append({
                            "bbox": {
                                "x": x1,
                                "y": y1,
                                "width": x2 - x1,
                                "height": y2 - y1
                            },
                            "confidence": float(confidence),
                            "model": "yolo"
                        })
            
            return {
                "faces_found": len(faces),
                "faces": faces,
                "image_shape": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "faces_found": 0,
                "faces": []
            }

    def detect_faces_enhanced(self, image_data):
        """
        Enhanced face detection using MediaPipe Face Mesh for precise detection
        including hair and ears
        """
        try:
            # Convert image data to numpy array
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # If it's already a PIL Image
                image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use Face Mesh for precise detection
            results = self.face_mesh.process(image_rgb)
            
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
                    
                    # Calculate extended bounding box to include hair and ears
                    x_coords = landmarks[:, 0]
                    y_coords = landmarks[:, 1]
                    
                    # Get basic face bounds
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    
                    # Extend bounds to include hair and ears
                    face_width = x_max - x_min
                    face_height = y_max - y_min
                    
                    # Extend horizontally for ears (about 20% on each side)
                    ear_extension = int(face_width * 0.2)
                    x_min = max(0, x_min - ear_extension)
                    x_max = min(image.shape[1], x_max + ear_extension)
                    
                    # Extend vertically for hair (about 30% on top)
                    hair_extension = int(face_height * 0.3)
                    y_min = max(0, y_min - hair_extension)
                    y_max = min(image.shape[0], y_max + int(face_height * 0.1))  # Small extension at bottom
                    
                    # Get key facial landmarks for reference
                    key_landmarks = self._extract_key_landmarks(landmarks)
                    
                    faces.append({
                        "bbox": {
                            "x": x_min,
                            "y": y_min,
                            "width": x_max - x_min,
                            "height": y_max - y_min
                        },
                        "confidence": 0.9,  # High confidence for mesh detection
                        "landmarks": landmarks.tolist(),
                        "key_landmarks": key_landmarks,
                        "face_region": {
                            "x": x_min,
                            "y": y_min,
                            "width": x_max - x_min,
                            "height": y_max - y_min
                        },
                        "model": "mediapipe_enhanced"
                    })
            
            return {
                "faces_found": len(faces),
                "faces": faces,
                "image_shape": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "faces_found": 0,
                "faces": []
            }

    def _extract_key_landmarks(self, landmarks):
        """
        Extract key facial landmarks for reference
        """
        # MediaPipe Face Mesh key landmark indices
        key_indices = {
            "nose_tip": 4,
            "left_eye_center": 33,
            "right_eye_center": 263,
            "left_ear": 234,
            "right_ear": 454,
            "mouth_left": 61,
            "mouth_right": 291,
            "chin": 152,
            "forehead": 10
        }
        
        key_landmarks = {}
        for name, idx in key_indices.items():
            if idx < len(landmarks):
                key_landmarks[name] = {
                    "x": int(landmarks[idx][0]),
                    "y": int(landmarks[idx][1])
                }
        
        return key_landmarks

    def detect_faces(self, image_data):
        """
        Original face detection method (kept for backward compatibility)
        """
        try:
            # Convert image data to numpy array
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # If it's already a PIL Image
                image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(image_rgb)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Get confidence score
                    confidence = detection.score[0]
                    
                    # Get key points (eyes, nose, mouth)
                    keypoints = {}
                    for keypoint in detection.location_data.relative_keypoints:
                        keypoints[f"kp_{len(keypoints)}"] = {
                            "x": keypoint.x * w,
                            "y": keypoint.y * h
                        }
                    
                    faces.append({
                        "bbox": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height
                        },
                        "confidence": float(confidence),
                        "keypoints": keypoints,
                        "model": "mediapipe_standard"
                    })
            
            return {
                "faces_found": len(faces),
                "faces": faces,
                "image_shape": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "faces_found": 0,
                "faces": []
            }

    def detect_faces_with_model(self, image_data, model: Literal["mediapipe_standard", "mediapipe_enhanced", "yolo"] = "mediapipe_enhanced"):
        """
        Detect faces using the specified model
        """
        if model == "yolo":
            return self.detect_faces_yolo(image_data)
        elif model == "mediapipe_enhanced":
            return self.detect_faces_enhanced(image_data)
        else:  # mediapipe_standard
            return self.detect_faces(image_data)

    def draw_faces_on_image(self, image_data, use_enhanced=True, model="mediapipe_enhanced"):
        """
        Draw bounding boxes around detected faces and return the annotated image
        """
        try:
            # Convert image data to numpy array
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if model == "yolo" and self.yolo_available:
                # Use YOLO detection
                results = self.yolo_model(image)
                
                # Draw YOLO detections
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            # Draw bounding box
                            cv2.rectangle(image_rgb, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        (0, 255, 0), 2)
                            
                            # Add confidence label
                            cv2.putText(image_rgb, 
                                      f"YOLO: {confidence:.2f}", 
                                      (int(x1), int(y1)-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0), 1)
                
                faces_found = sum(len(result.boxes) if result.boxes is not None else 0 for result in results)
                
            elif model == "mediapipe_enhanced":
                # Use enhanced detection
                results = self.face_mesh.process(image_rgb)
                
                # Draw enhanced detections
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Draw face mesh
                        self.mp_drawing.draw_landmarks(
                            image_rgb, 
                            face_landmarks, 
                            self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                        
                        # Draw bounding box
                        landmarks = []
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * image.shape[1])
                            y = int(landmark.y * image.shape[0])
                            landmarks.append([x, y])
                        
                        landmarks = np.array(landmarks)
                        x_coords = landmarks[:, 0]
                        y_coords = landmarks[:, 1]
                        
                        # Calculate extended bounds
                        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                        
                        face_width = x_max - x_min
                        face_height = y_max - y_min
                        
                        # Extend bounds
                        ear_extension = int(face_width * 0.2)
                        hair_extension = int(face_height * 0.3)
                        
                        x_min = max(0, x_min - ear_extension)
                        x_max = min(image.shape[1], x_max + ear_extension)
                        y_min = max(0, y_min - hair_extension)
                        y_max = min(image.shape[0], y_max + int(face_height * 0.1))
                        
                        # Draw extended bounding box
                        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.putText(image_rgb, "Enhanced Detection", (x_min, y_min-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                faces_found = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
                
            else:
                # Use original detection
                results = self.face_detection.process(image_rgb)
                
                # Draw original detections
                if results.detections:
                    for detection in results.detections:
                        self.mp_drawing.draw_detection(image_rgb, detection)
                
                faces_found = len(results.detections) if results.detections else 0
            
            # Convert back to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Convert to base64 for sending over API
            _, buffer = cv2.imencode('.jpg', image_bgr)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "annotated_image": f"data:image/jpeg;base64,{img_base64}",
                "faces_found": faces_found
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "annotated_image": None,
                "faces_found": 0
            } 