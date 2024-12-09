import os
import cv2
import json
from PIL import Image
import pillow_heif
import time
import re


# Register HEIC support
pillow_heif.register_heif_opener()

def create_dataset_from_bottle_folder(image_dir_path, dest_path, num_red, num_green, num_blue):
    # Define destination directories for images and labels
    image_dest = os.path.join(dest_path, "images")
    label_dest = os.path.join(dest_path, "labels")
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)

    # Iterate through all image files in the source directory
    for file_name in os.listdir(image_dir_path):
        image_path = os.path.join(image_dir_path, file_name)

        if file_name.endswith(".HEIC"):
            # Convert HEIC to JPG using PIL
            try:
                with Image.open(image_path) as img:
                    # Define new file name for the JPG image
                    time_in_ns = time.time_ns()
                    new_file_name = f"img_{time_in_ns}.jpg"
                    new_image_path = os.path.join(image_dest, new_file_name)
                    # Save as JPG
                    img.convert("RGB").save(new_image_path, "JPEG")
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                continue

            # Create label JSON
            label = {
                "image_id": new_file_name,
                "red_count": num_red,
                "green_count": num_green,
                "blue_count": num_blue
            }

            # Define the path to save the JSON label
            label_file_path = os.path.join(label_dest, new_file_name.replace(".jpg", ".json"))

            # Write the label to the label destination
            with open(label_file_path, 'w') as label_file:
                json.dump(label, label_file, indent=4)

            print(f"Processed and saved: {new_image_path}, {label_file_path}")


# Example usage
image_dir_path = "/Users/finnferchau/Downloads/finetuning_images"  # Replace with your HEIC folder path
dest_path = "/Users/finnferchau/dev/EH_Bottle_Dataset/finetune_dataset"  # Replace with your destination folder path

for image_dir in os.listdir(image_dir_path):
    image_path = os.path.join(image_dir_path, image_dir)
    if not re.match(r"^\d+x\d+x\d+$", image_dir):
        continue
    pellet_counts = image_dir.split("x")
    pellet_counts = [int(pellet_counts[i]) for i in range(len(pellet_counts))]
    num_red = pellet_counts[0]
    num_green = pellet_counts[1]
    num_blue = pellet_counts[2]
    create_dataset_from_bottle_folder(image_path, dest_path, num_red, num_green, num_blue)