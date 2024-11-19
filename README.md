# **Bottle Dataset with Labeled Pellet Counts**

## **Overview**

This dataset contains images of small bottles labeled with the corresponding number of red, green, and blue pellets inside. The images were captured against a plain background to ensure consistency.

Each image is associated with a label file containing the exact count of red, green, and blue pellets.

## **Dataset Structure**

The dataset is organized as follows:

    .
    ├── images/          	# .jpg files
    ├── labels/           	# .json labels for the images
    └── README.md

## **Labels Format**

Each JSON file in the labels/ folder contains:

```
{
  "red": <count>,   // Number of red pellets
  "green": <count>, // Number of green pellets
  "blue": <count>   // Number of blue pellets
}
```

### **Example Label**

For an image of a bottle with 5 red, 3 green, and 8 blue pellets:

```json
{
  "red": 5,
  "green": 3,
  "blue": 8
}
```

## **Statistics**

	•	Total Images: 400
	•	Total Labels: 400
	•	Bottle Variants: 8 different configurations (for now)
	•	Image Dimensions: 1280x720 pixels.

