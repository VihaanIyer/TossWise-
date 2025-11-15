"""
Object Detection Module using YOLOv8
Detects food items in hand from camera feed
"""

import cv2
from ultralytics import YOLO
import numpy as np


class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLOv8 model for object detection
        
        Args:
            model_path: Path to YOLOv8 model weights (default: yolov8n.pt)
        """
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
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
                    if confidence > 0.5:  # Person detected with good confidence
                        return True
        return False
    
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
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
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
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
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

