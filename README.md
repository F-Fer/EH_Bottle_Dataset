# **Bottle Dataset with Labeled Pellet Counts**

## **Overview**

This dataset contains images of small bottles labeled with the corresponding number of red, green, and blue pellets inside. The images were captured against a plain background to ensure consistency.

Each image is associated with a label file containing the exact count of red, green, and blue pellets.

## **Dataset Structure**

The dataset is organized as follows:

    .
    ├── images/          # .jpg files
    ├── labels/           # .json labels for the images
    └── README.md

## **Labels Format**

Each JSON file in the labels/ folder contains:

```
{
  "red_count": <count>,   // Number of red pellets
  "green_count": <count>, // Number of green pellets
  "blue_count": <count>   // Number of blue pellets
}
```

### **Example Label**

For an image of a bottle with 5 red, 3 green, and 8 blue pellets:

```json
{
  "red_count": 5,
  "green_count": 3,
  "blue_count": 8
}
```

## **Setup**

### **Environment**

- Artificial lighting with the window blinds closed

### **Camera Placement**

- The bottle is placed **right in front of** the wall
- The robot is placed **30cm** from the bottle
- Both are place **perpendicular** to the wall


<img src="IMG_7532.jpg" alt="IMG_7515" width="400">

<img src="IMG_7533.jpg" alt="IMG_7517" width="400">


## **Statistics**

- Total Images: 2500
- Total Labels: 2500
- Bottle Variants: 125 different configurations
- Image Dimensions: 1920x1080 pixels.

