import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('models/resnet50_model_v2.keras')
print("Model loaded...")

# Preprocessing function
def center_crop_and_resize(image, target_size=(224, 224)):
    """
    Crops the center square of an image and resizes it to the target size.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size for resizing (width, height).

    Returns:
        PIL.Image.Image: The cropped and resized image.
    """

    image = Image.fromarray(image)
    width, height = image.size

    # Calculate the cropping box for a center square
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    # Crop the center square
    image = image.crop((left, top, right, bottom))

    # Resize to the target size
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    return image # Add batch dimension

# Start the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 's' to take a picture and run inference, or 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert BGR (OpenCV default) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the live video feed
    cv2.imshow('Webcam', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # 's' key to take a picture
        # Save and preprocess the captured image
        preprocessed_image = center_crop_and_resize(frame)

        plt.imshow(preprocessed_image[0])
        plt.show()

        preprocess_image = np.expand_dims(preprocessed_image, axis=0)

        # Run inference
        predictions = model.predict(preprocessed_image)
        print("Predictions (Red, Green, Blue counts):", predictions[0])

    elif key == ord('q'):  # 'q' key to quit
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()