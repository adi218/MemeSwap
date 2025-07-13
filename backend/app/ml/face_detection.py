import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
import base64
import logging
from typing import Literal
from app.ml.config import ml_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetectionService:
    def __init__(self):
        logger.info("Initializing FaceDetectionService...")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe face detection and mesh
        logger.info("Initializing MediaPipe face detection...")
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=ml_settings.MEDIAPIPE_DETECTION_CONFIDENCE
        )
        
        # Enhanced face mesh for precise detection
        logger.info("Initializing MediaPipe face mesh...")
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=ml_settings.MAX_FACES_PER_IMAGE,
            refine_landmarks=True,
            min_detection_confidence=ml_settings.MEDIAPIPE_FACE_MESH_CONFIDENCE,
            min_tracking_confidence=ml_settings.MEDIAPIPE_FACE_MESH_CONFIDENCE
        )
        
        # Initialize YOLO face detection
        self.yolo_model = None
        self._load_yolo_model()
        
        # 3D face model points for pose estimation
        self._init_3d_face_model()
        logger.info("FaceDetectionService initialization complete!")

    def _load_yolo_model(self):
        """Load YOLO face detection model"""
        logger.info("Attempting to load YOLO model...")
        try:
            # YOLO face detection model path
            model_path = ml_settings.YOLO_MODEL_PATH
            
            # Try to load YOLO model if available
            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLO model from {model_path}...")
                self.yolo_model = YOLO(model_path)
                self.yolo_available = True
                logger.info("YOLO model loaded successfully!")
            except ImportError:
                logger.warning("Ultralytics not available, YOLO detection disabled")
                self.yolo_available = False
            except Exception as e:
                logger.error(f"Could not load YOLO model: {e}")
                self.yolo_available = False
                
        except Exception as e:
            logger.error(f"YOLO initialization error: {e}")
            self.yolo_available = False

    def detect_faces_yolo(self, image_data):
        """
        Detect faces using YOLO model
        """
        logger.info("Starting YOLO face detection...")
        if not self.yolo_available:
            logger.warning("YOLO model not available, returning error")
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

            logger.info(f"YOLO processing image of shape: {image.shape}")

            # Run YOLO detection
            results = self.yolo_model(image)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    logger.info(f"YOLO found {len(boxes)} potential faces")
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = box.conf[0].cpu().numpy()
                        
                        logger.info(f"YOLO face {i+1}: bbox=({x1},{y1},{x2},{y2}), confidence={confidence:.3f}")
                        
                        # Pose estimation using landmarks from the detected region
                        face_region = image[y1:y2, x1:x2]
                        pose_estimation = {"success": False}
                        
                        if face_region.size > 0:
                            face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                            mesh_results = self.face_mesh.process(face_region_rgb)
                            
                            if mesh_results.multi_face_landmarks:
                                face_landmarks = mesh_results.multi_face_landmarks[0]
                                landmarks = [(int(lm.x * face_region.shape[1]) + x1, int(lm.y * face_region.shape[0]) + y1) for lm in face_landmarks.landmark]
                                pose_estimation = self.estimate_face_pose(landmarks, image.shape)

                        faces.append({
                            "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                            "confidence": float(confidence),
                            "pose": pose_estimation,
                            "model": "yolo"
                        })
                else:
                    logger.info("YOLO found no faces")
            
            logger.info(f"YOLO detection complete: {len(faces)} faces found")
            return {
                "faces_found": len(faces),
                "faces": faces,
                "image_shape": {"width": image.shape[1], "height": image.shape[0]}
            }
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return {"error": str(e), "faces_found": 0, "faces": []}

    def detect_faces_enhanced(self, image_data):
        """
        Enhanced face detection using MediaPipe Face Mesh for precise detection.
        This version provides a tighter bounding box around the face.
        """
        logger.info("Starting enhanced MediaPipe face detection...")
        try:
            # Convert image data to numpy array
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

            logger.info(f"Enhanced MediaPipe processing image of shape: {image.shape}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            faces = []
            if results.multi_face_landmarks:
                logger.info(f"Enhanced MediaPipe found {len(results.multi_face_landmarks)} faces")
                for i, face_landmarks in enumerate(results.multi_face_landmarks):
                    landmarks = np.array([(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in face_landmarks.landmark])
                    
                    x_min, y_min = np.min(landmarks, axis=0)
                    x_max, y_max = np.max(landmarks, axis=0)
                    
                    # Refined bounding box to be tighter around the face
                    face_width = x_max - x_min
                    face_height = y_max - y_min
                    
                    # A more conservative extension
                    x_min = max(0, x_min - int(face_width * 0.1))
                    x_max = min(image.shape[1], x_max + int(face_width * 0.1))
                    y_min = max(0, y_min - int(face_height * 0.2)) # A bit more for forehead/hair
                    y_max = min(image.shape[0], y_max + int(face_height * 0.05))

                    key_landmarks = self._extract_key_landmarks(landmarks)
                    pose_estimation = self.estimate_face_pose(landmarks, image.shape)
                    
                    faces.append({
                        "bbox": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
                        "confidence": 0.9,
                        "landmarks": landmarks.tolist(),
                        "key_landmarks": key_landmarks,
                        "pose": pose_estimation,
                        "model": "mediapipe_enhanced"
                    })
            else:
                logger.info("Enhanced MediaPipe found no faces")
            
            return {
                "faces_found": len(faces),
                "faces": faces,
                "image_shape": {"width": image.shape[1], "height": image.shape[0]}
            }
            
        except Exception as e:
            logger.error(f"Enhanced MediaPipe detection error: {e}")
            return {"error": str(e), "faces_found": 0, "faces": []}

    def _extract_key_landmarks(self, landmarks):
        """Extract key facial landmarks from the full landmark set."""
        key_indices = {
            "nose_tip": 4, "left_eye_center": 33, "right_eye_center": 263,
            "left_ear": 234, "right_ear": 454, "mouth_left": 61,
            "mouth_right": 291, "chin": 152, "forehead": 10
        }
        return {name: {"x": int(landmarks[idx][0]), "y": int(landmarks[idx][1])} 
                for name, idx in key_indices.items() if idx < len(landmarks)}

    def detect_faces(self, image_data):
        """Original face detection method for backward compatibility."""
        logger.info("Starting standard MediaPipe face detection...")
        try:
            # Convert image data to numpy array
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            faces = []
            if results.detections:
                h, w, _ = image.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                    
                    # Pose estimation logic remains the same
                    pose_estimation = {"success": False}
                    face_region = image[y:y+height, x:x+width]
                    if face_region.size > 0:
                        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                        mesh_results = self.face_mesh.process(face_region_rgb)
                        if mesh_results.multi_face_landmarks:
                            landmarks = [(int(lm.x * width) + x, int(lm.y * height) + y) for lm in mesh_results.multi_face_landmarks[0].landmark]
                            pose_estimation = self.estimate_face_pose(landmarks, image.shape)
                    
                    faces.append({
                        "bbox": {"x": x, "y": y, "width": width, "height": height},
                        "confidence": float(detection.score[0]),
                        "pose": pose_estimation,
                        "model": "mediapipe_standard"
                    })
            return {
                "faces_found": len(faces),
                "faces": faces,
                "image_shape": {"width": image.shape[1], "height": image.shape[0]}
            }
        except Exception as e:
            logger.error(f"Standard MediaPipe detection error: {e}")
            return {"error": str(e), "faces_found": 0, "faces": []}

    def detect_faces_with_model(self, image_data, model: Literal["mediapipe_standard", "mediapipe_enhanced", "yolo"] = "mediapipe_enhanced"):
        """Detect faces using the specified model."""
        logger.info(f"Detecting faces using model: {model}")
        if model == "yolo":
            return self.detect_faces_yolo(image_data)
        elif model == "mediapipe_enhanced":
            return self.detect_faces_enhanced(image_data)
        else:
            return self.detect_faces(image_data)

    def draw_faces_on_image(self, image_data, model="mediapipe_enhanced"):
        """Draw bounding boxes and landmarks on the image."""
        try:
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

            detection_result = self.detect_faces_with_model(image, model)
            
            if "error" in detection_result:
                logger.error(f"Could not draw faces: {detection_result['error']}")
                return {"error": detection_result['error'], "annotated_image": None, "faces_found": 0}

            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            for face in detection_result["faces"]:
                bbox = face["bbox"]
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # Draw bounding box
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label
                label = f"{face['model']}: {face['confidence']:.2f}"
                cv2.putText(image_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Optionally draw landmarks for enhanced model
                if model == "mediapipe_enhanced" and "landmarks" in face:
                    for landmark in face["landmarks"]:
                        cv2.circle(image_rgb, tuple(landmark), 1, (0, 0, 255), -1)

            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "annotated_image": f"data:image/jpeg;base64,{img_base64}",
                "faces_found": detection_result["faces_found"]
            }
            
        except Exception as e:
            logger.error(f"Error drawing faces: {e}")
            return {"error": str(e), "annotated_image": None, "faces_found": 0}

    def _init_3d_face_model(self):
        """Initialize 3D face model points for pose estimation."""
        self.face_3d_model = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye left corner
            [225.0, 170.0, -135.0],    # Right eye right corner
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0, -150.0, -125.0]    # Right mouth corner
        ], dtype=np.float64)

    def estimate_face_pose(self, landmarks, image_shape):
        """Estimate face pose (yaw, pitch, roll) from facial landmarks."""
        try:
            landmarks = np.array(landmarks, dtype=np.float64)
            pose_landmark_indices = [1, 152, 226, 446, 61, 291]
            
            if landmarks.shape[0] < max(pose_landmark_indices) + 1:
                return {"success": False, "error": "Not enough landmarks for pose estimation"}

            face_2d = landmarks[pose_landmark_indices]
            
            h, w = image_shape[:2]
            focal_length = w
            camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))
            
            success, rotation_vec, _ = cv2.solvePnP(self.face_3d_model, face_2d, camera_matrix, dist_coeffs)
            
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
                
                singular = sy < 1e-6
                if not singular:
                    pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                else:
                    pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                    yaw = 0
                    roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])

                return {
                    "yaw": np.degrees(yaw), "pitch": np.degrees(pitch), "roll": np.degrees(roll),
                    "confidence": self._calculate_pose_confidence(landmarks), "success": True
                }
            else:
                return {"success": False, "error": "solvePnP failed"}
        except Exception as e:
            logger.error(f"Pose estimation error: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_pose_confidence(self, landmarks):
        """Calculate a confidence score for the pose estimation based on landmark spread."""
        try:
            x_coords, y_coords = landmarks[:, 0], landmarks[:, 1]
            spread = np.std(x_coords) + np.std(y_coords)
            
            # Normalize spread to a confidence score (heuristic)
            confidence = min(1.0, spread / 100.0) 
            return confidence
        except Exception:
            return 0.5