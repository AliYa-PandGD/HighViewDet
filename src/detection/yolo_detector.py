"""
YOLO Detector Wrapper - Supports YOLOv3, v4, v5, v8, v9, and v26+
Provides a unified interface for all YOLO versions
Supports both online (auto-download) and local file loading
"""

import numpy as np
import cv2
import os
from ultralytics import YOLO
from config import MIN_CONF


class YOLODetector:
    """
    Unified YOLO detector that supports multiple YOLO versions
    """
    
    def __init__(self, model_name="yolov8n", device=0, conf_threshold=MIN_CONF):
        """
        Initialize YOLO detector
        
        Args:
            model_name (str): YOLO model to load. Can be:
                
                A) Model name (auto-download):
                - 'yolov3', 'yolov3-tiny'
                - 'yolov4', 'yolov4-tiny'
                - 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
                - 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
                - 'yolov9n', 'yolov9s', 'yolov9m', 'yolov9c', 'yolov9e'
                - 'yolov26n', 'yolov26s', 'yolov26m', etc.
                
                B) Local file path (e.g., 'models/yolov8n.pt' or 'D:/path/to/model.pt')
                
            device (int or str): GPU device ID (0, 1, etc.) or 'cpu'
            conf_threshold (float): Confidence threshold for detection
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Check if model_name is a local file path
        if os.path.isfile(model_name):
            print(f"Loading model from local file: {model_name}...")
            model_path = model_name
        else:
            # Treat as model name for auto-download
            print(f"Loading model: {model_name}...")
            model_path = f"{model_name}.pt"
        
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"Model loaded successfully on device: {device}")

    
    def detect(self, frame):
        """
        Detect persons in frame
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (boxes, confidences, centroids) where:
                - boxes: List of [x, y, width, height]
                - confidences: List of confidence scores
                - centroids: List of (center_x, center_y) tuples
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        boxes = []
        confidences = []
        centroids = []
        
        # Process detections (class_id 0 is 'person' in COCO dataset)
        for result in results:
            for detection in result.boxes:
                # Get class ID and confidence
                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])
                
                # Only process 'person' class (class_id == 0)
                if class_id == 0 and confidence > self.conf_threshold:
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = detection.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Convert to (x, y, width, height) format
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width // 2
                    center_y = y1 + height // 2
                    
                    boxes.append([x1, y1, width, height])
                    confidences.append(confidence)
                    centroids.append((center_x, center_y))
        
        return boxes, confidences, centroids


def create_yolo_detector(yolo_version="yolov8n", device=0, conf_threshold=MIN_CONF):
    """
    Factory function to create YOLO detector
    
    Args:
        yolo_version (str): YOLO model version
        device (int or str): Device to run on ('cpu' or GPU device ID)
        conf_threshold (float): Confidence threshold
        
    Returns:
        YOLODetector: Initialized detector
    """
    return YOLODetector(model_name=yolo_version, device=device, conf_threshold=conf_threshold)
