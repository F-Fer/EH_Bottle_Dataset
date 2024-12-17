import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt


def extract_bottle_bottom(image, model, confidence_threshold=0.5, crop_ratio=1.0, target_size=224):
    # Perform inference
    results = model.predict(source=image, conf=confidence_threshold, verbose=False)
    detections = results[0]

    if detections.boxes is None or len(detections.boxes) == 0:
        print(f"No bottles detected in image.")
        return None, None

    # Get bounding box of the first detected object
    box = detections.boxes[0].xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = map(int, box)

    # Calculate width and height of the bounding box
    box_width = x_max - x_min
    box_height = y_max - y_min

    # Calculate the center of the bounding box
    center_x = x_min + box_width // 2
    center_y = y_min + box_height // 2

    # Calculate the side length of the square crop
    crop_size = max(box_width, box_height, target_size)

    # Determine the square crop boundaries
    crop_x_min = max(0, center_x - crop_size // 2)
    crop_y_min = max(0, center_y - crop_size // 2)
    crop_x_max = crop_x_min + crop_size
    crop_y_max = crop_y_min + crop_size

    # Adjust crop to ensure it fits within the image boundaries
    if crop_x_max > image.shape[1]:
        crop_x_min -= (crop_x_max - image.shape[1])
        crop_x_max = image.shape[1]
    if crop_y_max > image.shape[0]:
        crop_y_min -= (crop_y_max - image.shape[0])
        crop_y_max = image.shape[0]

    # Ensure the adjustments keep the crop within valid bounds
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)

    # Perform cropping
    cropped_bottle = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # Resize the cropped image to the target size (224x224)
    resized_bottle = cv2.resize(cropped_bottle, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized_bottle, [x_min, y_min, x_max, y_max]


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
keras_path = "/Users/finnferchau/dev/EH_Bottle_Dataset/models/resnet50_model_v21.keras"

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
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if not ret:
        print("Error: Unable to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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