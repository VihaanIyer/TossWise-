"""
Object Detection Module using YOLOv8 and Roboflow
Detects trash items from camera feed
"""

import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Warning: roboflow package not available, trash detection will be disabled")


class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLOv8 model for object detection and Roboflow client
        
        Args:
            model_path: Path to YOLOv8 model weights (default: yolov8n.pt)
        """
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("YOLOv8 model loaded successfully!")
        
        # Initialize Roboflow client for trash detection
        print("Initializing Roboflow client...")
        self.roboflow_api_key = "i34YBNJrrsWv3HEBIeNs"
        self.roboflow_model_id = "yolov8-trash-detections/6"
        self.roboflow_client = None
        
        if ROBOFLOW_AVAILABLE:
            try:
                # Initialize Roboflow client
                rf = Roboflow(api_key=self.roboflow_api_key)
                # Parse model_id: "yolov8-trash-detections/6" -> workspace/project/version
                # We need to get the workspace - try to infer from model
                # For now, we'll use the model directly via the roboflow package
                # The model_id format suggests: project/version
                # We'll need to know the workspace, but let's try a different approach
                self.roboflow_client = rf
                self.roboflow_workspace = None  # Will be determined from API
                print("Roboflow client initialized successfully!")
            except Exception as e:
                print(f"Warning: Could not initialize Roboflow client: {e}")
                self.roboflow_client = None
        else:
            print("Roboflow package not available")
        
        # Common food-related classes in COCO dataset
        self.food_classes = [
            'apple', 'banana', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'bottle', 'cup', 'bowl', 'sandwich',
            'orange', 'pizza', 'donut', 'cake'
        ]
        
        # Classes to exclude (people, animals, furniture, etc.)
        # NOTE: Removed trash items like toothbrush, hair drier, scissors from excluded list
        self.excluded_classes = [
            'person', 'people', 'man', 'woman', 'child', 'boy', 'girl',
            'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'book', 'clock', 'vase', 'chair', 'couch',
            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'keyboard', 'remote', 'monitor'
        ]
        
        # Trash/waste related items to detect (expanded list)
        self.trash_classes = [
            'bottle', 'cup', 'bowl', 'apple', 'banana', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'sandwich', 'fork', 'knife', 'spoon', 'banana', 'apple',
            'bottle', 'wine glass', 'cup', 'bowl', 'banana', 'apple',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'sandwich', 'cell phone', 'remote', 'keyboard',
            'mouse', 'laptop', 'book', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush', 'tv', 'monitor', 'keyboard'
        ]
        
        # Also detect common objects that could be trash
        # Lower the threshold for detection
    
    def detect_person(self, frame):
        """
        Detect if a person is present in the frame
        Returns True if person detected, False otherwise
        """
        results = self.model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                # Check if it's a person with reasonable confidence
                if class_name.lower() in ['person', 'people', 'man', 'woman', 'child', 'boy', 'girl']:
                    if confidence > 0.4:  # Lower threshold for faster detection (was 0.5)
                        return True
        return False
    
    def detect_trash_objects(self, frame):
        """
        Detect trash objects using Roboflow YOLOv8 model
        Returns list of detected trash objects, or empty list if none found
        
        Args:
            frame: Input image frame (numpy array in BGR format)
            
        Returns:
            List of detected trash objects with their classes and confidence scores
        """
        if self.roboflow_client is None:
            return []
        
        try:
            # Convert BGR frame to RGB PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize image if too large (Roboflow API has size limits)
            # Use smaller size for faster API calls and less lag
            max_size = 1280  # Reduced from 1920 for faster processing
            if pil_image.width > max_size or pil_image.height > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Parse model_id: "yolov8-trash-detections/6" = project/version
            project_name = self.roboflow_model_id.split('/')[0]
            version_num = int(self.roboflow_model_id.split('/')[1])
            
            # Try to access the project directly
            # The roboflow package requires workspace/project/version format
            # Since we only have project/version, we need to try accessing it directly
            # or the user needs to provide the workspace name
            try:
                # Try accessing project directly (works if it's in your default workspace)
                # The roboflow package may auto-detect workspace for your projects
                project = self.roboflow_client.project(project_name)
                model = project.version(version_num).model
                
                # Save PIL image to temporary file for Roboflow (it expects file path)
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    pil_image.save(tmp_file.name, format='JPEG')
                    temp_path = tmp_file.name
                
                try:
                    # Run prediction with file path
                    # Use lower confidence threshold for faster detection, higher overlap for better accuracy
                    prediction = model.predict(temp_path, confidence=30, overlap=30)
                    result = prediction.json()
                finally:
                    # Clean up temporary file immediately
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
            except Exception as e1:
                # If direct access fails, the project might be in a specific workspace
                # Try to get workspace from the model_id or use a default
                print(f"Direct project access failed: {e1}")
                print(f"Note: Roboflow requires workspace name. Model ID format should be: workspace/project/version")
                print(f"      Current model_id: {self.roboflow_model_id}")
                print(f"      If you know the workspace, update model_id to: workspace/{project_name}/{version_num}")
                return []
            
            # Parse results
            detections = []
            if result:
                # Handle different response formats
                predictions = []
                if isinstance(result, dict):
                    if 'predictions' in result:
                        predictions = result['predictions']
                    elif 'results' in result:
                        predictions = result['results']
                elif isinstance(result, list):
                    predictions = result
                
                for pred in predictions:
                    if isinstance(pred, dict):
                        class_name = pred.get('class', '') or pred.get('class_name', '')
                        confidence = pred.get('confidence', 0.0) or pred.get('confidence_score', 0.0)
                        
                        # Get bounding box coordinates
                        if 'x' in pred and 'y' in pred:
                            # Center format (x, y, width, height)
                            x = pred.get('x', 0)
                            y = pred.get('y', 0)
                            width = pred.get('width', 0)
                            height = pred.get('height', 0)
                            
                            x1 = int(x - width / 2)
                            y1 = int(y - height / 2)
                            x2 = int(x + width / 2)
                            y2 = int(y + height / 2)
                        elif 'x1' in pred:
                            # Corner format
                            x1 = int(pred.get('x1', 0))
                            y1 = int(pred.get('y1', 0))
                            x2 = int(pred.get('x2', 0))
                            y2 = int(pred.get('y2', 0))
                        else:
                            continue
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error in Roboflow trash detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_objects(self, frame, filter_trash_only=True):
        """
        Detect objects in the frame, optionally filtering for trash/food items only
        
        Args:
            frame: Input image frame (numpy array)
            filter_trash_only: If True, only return trash/food items (exclude people)
            
        Returns:
            List of detected objects with their classes and confidence scores
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        # Frame geometry for relative area checks
        h, w = frame.shape[:2]
        frame_area = float(w * h)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_area = float((x2 - x1) * (y2 - y1))
                
                # -------------------------------
                # SMART GLOVE FILTER
                # -------------------------------
                if class_name.lower() == "glove":
                    # Tune these as needed
                    MIN_CONF_GLOVE = 0.80     # require high confidence
                    MAX_REL_AREA_GLOVE = 0.30 # gloves shouldn't cover 30%+ of frame
                    is_low_conf = confidence < MIN_CONF_GLOVE
                    is_too_big = box_area > MAX_REL_AREA_GLOVE * frame_area
                    # If it's low confidence OR absurdly large, assume it's a hand
                    if is_low_conf or is_too_big:
                        # Skip this detection entirely
                        continue
                # -------------------------------
                # END GLOVE FILTER
                # -------------------------------
                
                # Filter out excluded classes (people, etc.)
                if filter_trash_only:
                    class_name_lower = class_name.lower()
                    # Skip if it's an excluded class
                    if any(excluded in class_name_lower for excluded in self.excluded_classes):
                        continue
                    # Include trash/food related items OR any object with decent confidence
                    # (more permissive to catch items in hand)
                    is_trash_item = any(trash in class_name_lower for trash in self.trash_classes)
                    # If it's not a trash item but has high confidence (>60%), include it anyway
                    # (might be something we should classify)
                    if not is_trash_item:
                        if confidence < 0.6:
                            continue
                        # High confidence non-trash item - might be trash, include it
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return detections
    
    def detect_food_in_hand(self, frame):
        """
        Detect food items in the frame, focusing on items that might be in hand
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected food items
        """
        # Use filtered detection (already excludes people)
        detections = self.detect_objects(frame, filter_trash_only=True)
        
        # Filter for items with high confidence
        food_items = []
        for detection in detections:
            if detection['confidence'] > 0.5:  # Confidence threshold
                food_items.append(detection)
        
        return food_items
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input image frame
            detections: List of detections to draw
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Choose color based on confidence (green for high, yellow for medium)
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw bounding box with thicker lines
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_height - 10), 
                         (x1 + label_width + 5, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return annotated_frame

