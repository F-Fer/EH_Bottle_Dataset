import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt


def extract_bottle_bottom(image, model, confidence_threshold=0.5, crop_ratio=0.6, target_size=(224, 224)):
    # Perform YOLO inference
    results = model.predict(source=image, conf=confidence_threshold, verbose=False)
    detections = results[0]

    # Skip if no detections are found
    if detections.boxes is None or len(detections.boxes) == 0:
        print("No bottles detected.")
        return None, None

    # Get bounding box of the first detected object
    box = detections.boxes[0].xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = map(int, box)

    # Calculate the bottom portion
    bottle_height = y_max - y_min
    bottom_y_min = y_max - int(bottle_height * crop_ratio)

    # Ensure the crop is within the image bounds
    bottom_y_min = max(0, bottom_y_min)

    # Crop the bottom portion
    cropped_bottom = image[bottom_y_min:y_max, x_min:x_max]

    # Resize to the target size
    resized_cropped_bottom = cv2.resize(cropped_bottom, target_size, interpolation=cv2.INTER_AREA)

    return resized_cropped_bottom, (x_min, y_min, x_max, y_max)


def preprocess_image(image, add_batch_dim=True, target_size=(224, 224)):
    # Convert to PIL Image
    image = Image.fromarray(image)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = np.array(image) / 255.0

    if add_batch_dim:
        image = np.expand_dims(image, axis=0)  # Add batch dimension

    return image


def run_inference_on_bottle(image, keras_model, yolo_model):
    cropped_bottle, bbox = extract_bottle_bottom(image, yolo_model)
    if cropped_bottle is None:
        raise ValueError("Could not detect bottle in image.")

    preprocessed_image = preprocess_image(cropped_bottle)
    predictions = keras_model.predict(preprocessed_image)
    return predictions, bbox


# Paths to models
image_dir_path = "/Users/finnferchau/dev/EH_Bottle_Dataset/images"
yolo_path = "/Users/finnferchau/dev/EH_Bottle_Dataset/models/yolo.pt"
keras_path = "/Users/finnferchau/dev/EH_Bottle_Dataset/models/resnet50_model_v14.keras"

# Load models
keras_model = tf.keras.models.load_model(keras_path)
yolo_model = YOLO(yolo_path)
print("Loaded models ...")

# Start the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 's' to take a picture and run inference, or 'q' to quit.")

while True:
    # # Capture frame-by-frame
    # ret, frame = cap.read()
    # if not ret:
    #     print("Error: Unable to capture frame.")
    #     break

    # # Perform YOLO inference and draw bounding boxes
    # results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
    # detections = results[0]

    # if detections.boxes is not None and len(detections.boxes) > 0:
    #     for box in detections.boxes:
    #         # Extract bounding box coordinates
    #         xyxy = box.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
    #         x_min, y_min, x_max, y_max = map(int, xyxy)

    #         # Draw the bounding box on the frame
    #         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    #         label = f"Confidence: {box.conf[0]:.2f}"
    #         cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # # Display the live video feed
    # cv2.imshow('Webcam with Detections', frame)

    # Wait for a key press
    # key = cv2.waitKey(1) & 0xFF

    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if not ret:
        print("Error: Unable to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform YOLO inference and draw bounding boxes
    # results = yolo_model.predict(source=frame_rgb, conf=0.5, verbose=False)
    # detections = results[0]

    try:
        predictions, bbox = run_inference_on_bottle(frame_rgb, keras_model=keras_model, yolo_model=yolo_model)
        print(f"Predictions: R: {predictions[0][0]}, G: {predictions[0][1]}, B: {predictions[0][2]}")

        # Highlight the detected bottle in the frame
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            label = f"R: {int(predictions[0][0])}, G: {int(predictions[0][1])}, B: {int(predictions[0][2])}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Webcam with Detections', frame)
    except ValueError as e:
        print(str(e))
    
    # Display the live video feed
    cv2.imshow('Webcam with Detections', frame)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()