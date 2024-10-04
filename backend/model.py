import cv2
import numpy as np
from ultralytics import YOLO

# Random colors for each class (you can modify these for your dataset)
COLORS = np.random.uniform(0, 255, size=(80, 3))  # Assuming 80 classes, adjust for your dataset

def predict(image: np.ndarray, task: str) -> np.ndarray:
    if task == 'detection':
        # Load the YOLOv8 detection model
        model = YOLO('models/yolov8n.pt')  # Adjust to your model's path
        results = model.predict(image)
        
        # Get the detection results
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_names = model.names
        
        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image, boxes, class_ids, scores, class_names)
        return image_with_boxes

    elif task == 'segmentation':
        # Load the YOLOv8 segmentation model
        model = YOLO('models/yolov8n-seg.pt')  # Adjust to your model's path
        results = model.predict(image)

        # Extract segmentation masks, class ids, and scores
        masks = results[0].masks.xy  # List of polygon coordinates for each mask
        class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs for each mask
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_names = model.names
        
        # Draw segmentation masks on the image with class colors and labels
        image_with_masks = draw_masks(image, masks, class_ids, scores, class_names)
        return image_with_masks


def draw_boxes(image: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray, scores: np.ndarray, class_names: list) -> np.ndarray:
    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(class_id)]}: {score:.2f}"

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)

        # Calculate the label size and position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=3)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # Draw the label background rectangle
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color=(255, 255, 255), thickness=cv2.FILLED)

        # Draw the label
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=3)

    return image


def draw_masks(image: np.ndarray, masks: list, class_ids: np.ndarray, scores: np.ndarray, class_names: list) -> np.ndarray:
    for mask, class_id, score in zip(masks, class_ids, scores):
        # The mask is a polygon represented by a list of xy coordinates
        polygon = np.array(mask, dtype=np.int32)

        # Choose a color based on the class ID
        color = COLORS[int(class_id)]

        # Fill the polygon with the corresponding color
        cv2.fillPoly(image, [polygon], color=color)

        # Calculate the centroid of the mask (average of polygon points)
        centroid = np.mean(polygon, axis=0).astype(int)

        # Prepare the label
        label = f"{class_names[int(class_id)]}: {score:.2f}"

        # Get the size of the label text
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=3)

        # Calculate text position to center it on the mask
        text_x = int(centroid[0] - text_size[0] / 2)
        text_y = int(centroid[1] + text_size[1] / 2)

        # Add the label centered on the mask
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=3)

    return image

