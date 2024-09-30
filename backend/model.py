import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (pretrained)
model = YOLO('yolov8n.pt')  # Adjust to your model's path

def predict(image: np.ndarray) -> np.ndarray:
    # Perform inference with YOLOv8
    results = model.predict(image)

    # Get the results (bounding boxes, class ids, scores)
    boxes = results[0].boxes.xyxy.cpu().numpy()   # Bounding box coordinates
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs
    scores = results[0].boxes.conf.cpu().numpy()    # Confidence scores

    # Load class names (COCO dataset classes, or use your custom classes)
    class_names = model.names

    # Draw bounding boxes and labels on the image
    image_with_boxes = draw_boxes(image, boxes, class_ids, scores, class_names)

    # Return the image with overlaid bounding boxes
    return image_with_boxes

def draw_boxes(image: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray, scores: np.ndarray, class_names: list) -> np.ndarray:
    # Iterate through all boxes, class ids, and scores
    for box, class_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
        label = f"{class_names[int(class_id)]}: {score:.2f}"

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Calculate the label size and position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10  # Avoid text going out of bounds

        # Draw the label background rectangle
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color=(0, 255, 0), thickness=cv2.FILLED)

        # Draw the label
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

    return image
