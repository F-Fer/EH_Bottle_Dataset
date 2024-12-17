# **Bottle Dataset with Labeled Pellet Counts**

## **Overview**

This dataset contains images of small bottles labeled with the corresponding number of red, green, and blue pellets inside. The majority of the images were captured in frot of a white wall with white ground. However the dataset also include about 300 additional images in a veriety of environments.

Each image is associated with a label file containing the exact count of red, green, and blue pellets.

## **Dataset Structure**

The dataset is organized as follows:

    .
    ├── images_raw/          # .jpg files
    ├── images_cropped/      # .jpg images cropped to 224x224
    ├── labels/           	 # .json labels for the images
    └── README.md

## **Labels Format**

Each JSON file in the labels/ folder contains:

```
{
  "image_id": <image file name>,
  "red": <count>,   // Number of red pellets
  "green": <count>, // Number of green pellets
  "blue": <count>   // Number of blue pellets
}
```

### **Example Label**

For an image of a bottle with 5 red, 3 green, and 8 blue pellets:

```json
{
    "image_id": "img_20241125_082521.jpg",
    "red": 5,
    "green": 3,
    "blue": 8
}
```

## **Statistics**

	•	Total Images: 2,967
	•	Total Labels: 2,967
	•	Bottle Variants: 125 variants

