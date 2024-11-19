# Bottle Dataset with Labeled Pallet Counts

## Overview

This dataset contains images of small bottles labeled with the corresponding number of red, green, and blue pallets inside. The images were captured against a plain background to ensure consistency and ease of use in machine learning and computer vision applications.

The dataset is suitable for tasks such as:
	•	Object detection and classification
	•	Pallet counting based on color
	•	Training models for robotic perception tasks

Each image is associated with a label file containing the exact count of red, green, and blue pallets.

## Dataset Structure

The dataset is organized as follows:

dataset/
├── images/         # Contains all bottle images
├── labels/         # Contains JSON files with the pallet counts
├── README.md       # Description of the dataset
└── LICENSE         # Licensing details

## Labels Format

Each JSON file in the labels/ folder contains:

{
  "red": <count>,   // Number of red pallets
  "green": <count>, // Number of green pallets
  "blue": <count>   // Number of blue pallets
}

### Example Label

For an image of a bottle with 5 red, 3 green, and 8 blue pallets:

{
  "red": 5,
  "green": 3,
  "blue": 8
}

Usage

	1.	Download the Dataset
Clone the repository or download the dataset from GitHub.
	2.	Load Images and Labels
Example Python code to load the dataset:

import os
import json
from PIL import Image

images_dir = "path/to/images"
labels_dir = "path/to/labels"

# Example: Load an image and its corresponding label
image_file = os.path.join(images_dir, "bottle10x10x10_1.jpg")
label_file = os.path.join(labels_dir, "bottle10x10x10_1.json")

image = Image.open(image_file)
with open(label_file, 'r') as f:
    label = json.load(f)

print("Red:", label["red"], "Green:", label["green"], "Blue:", label["blue"])


	3.	Train Your Model
Use the dataset for object detection, classification, or other tasks.

Statistics

	•	Total Images: X
	•	Total Labels: X
	•	Bottle Variants: X different configurations
	•	Image Dimensions: Consistent resolution, e.g., 1280x720 pixels.

License

This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
You are free to:
	•	Share — copy and redistribute the material in any medium or format
	•	Adapt — remix, transform, and build upon the material for any purpose, even commercially

As long as you provide appropriate credit. See the LICENSE file for more details.

Citation

If you use this dataset, please cite:

Finn Ferchau. (2024). Bottle Dataset with Labeled Pallet Counts.

Acknowledgments

Special thanks to the ROS project team for providing the setup and support during dataset creation.

Let me know if you’d like assistance filling in the missing details (e.g., total images or GitHub repository link).