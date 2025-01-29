import cv2
import numpy as np
import torch
from ultralytics import YOLO
from prometheus_client import Counter, Summary

# Random colors for each class (you can modify these for your dataset)
COLORS = np.random.uniform(0, 255, size=(80, 3))  # Assuming 80 classes, adjust for your dataset

# Prometheus metrics for prediction counts and latency
PREDICTION_COUNT = Counter('prediction_count_total', 'Total number of predictions', ['task'])
PREDICTION_LATENCY = Summary('prediction_latency_seconds', 'Time taken for a prediction', ['task'])

def predict(image: np.ndarray, task: str) -> np.ndarray:
    if task == 'detection':
        with PREDICTION_LATENCY.labels(task='detection').time():  # Measure prediction latency
            # Load the YOLOv8 detection model
            model = YOLO('models/yolov8n.pt')  # Adjust to your model's path
            if torch.cuda.is_available():
                print("Using CUDA")
                results = model.predict(image, device='cuda')
            results = model.predict(image)
            # Get the detection results
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_names = model.names
            
            # Draw bounding boxes on the image
            image_with_boxes = draw_boxes(image, boxes, class_ids, scores, class_names)
            
            # Increment prediction count for detection
            PREDICTION_COUNT.labels(task='detection').inc()
            
            return image_with_boxes

    elif task == 'segmentation':
        with PREDICTION_LATENCY.labels(task='segmentation').time():  # Measure prediction latency
            # Load the YOLOv8 segmentation model
            model = YOLO('models/yolov8n-seg.pt')  # Adjust to your model's path
            if torch.cuda.is_available():
                print("Using CUDA")
                results = model.predict(image, device='cuda')
            results = model.predict(image)

            # Extract segmentation masks, class ids, and scores
            masks = results[0].masks.xy  # List of polygon coordinates for each mask
            class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs for each mask
            scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            class_names = model.names
            
            # Draw segmentation masks on the image with class colors and labels
            image_with_masks = draw_masks(image, masks, class_ids, scores, class_names)
            
            # Increment prediction count for segmentation
            PREDICTION_COUNT.labels(task='segmentation').inc()
            
            return image_with_masks

def draw_boxes(image: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray, scores: np.ndarray, class_names: list) -> np.ndarray:
    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(class_id)]}: {score:.2f}"

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)

        # Calculate the label size and position for the bottom-left corner
        font_scale = 0.6  # Smaller font scale for the label
        thickness = 1  # Thinner text thickness
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_x = x1 + 5  # Offset a little from the left edge
        text_y = y2 - 5  # Slight offset from the bottom

        # Draw the label text inside the box, bottom-left
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    
    return image

def draw_masks(image: np.ndarray, masks: list, class_ids: np.ndarray, scores: np.ndarray, class_names: list) -> np.ndarray:
    for mask, class_id, score in zip(masks, class_ids, scores):
        # The mask is a polygon represented by a list of xy coordinates
        polygon = np.array(mask, dtype=np.int32)

        # Choose a color based on the class ID
        color = COLORS[int(class_id)]

        # Fill the polygon with the corresponding color
        cv2.fillPoly(image, [polygon], color=color)

        # Get bounding box coordinates around the mask
        x1, y1, x2, y2 = cv2.boundingRect(polygon)

        # Prepare the label
        label = f"{class_names[int(class_id)]}: {score:.2f}"

        # Get the size of the label text
        font_scale = 1.2
        thickness = 2
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]

        # Calculate text position for the bottom-left corner of the mask bounding box
        text_x = x1 + 5  # Offset from left
        text_y = y2 - 5  # Offset from bottom

        # Add the label inside the mask at the bottom-left corner
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)

    return image
