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

### **Robot Placement**

- The bottle is placed **40cm** from the wall
- The robot is placed **27cm** from the bottle
- Both are place **perpendicular to the wall**


<img src="https://github.com/user-attachments/assets/3e494353-e617-4bb2-b3a3-59a2de734759" alt="IMG_7515" width="400">

<img src="https://github.com/user-attachments/assets/b9d2d648-d3f4-4a65-ab2c-28485ed8045a" alt="IMG_7517" width="400">


## **Statistics**

- Total Images: 400
- Total Labels: 400
- Bottle Variants: 8 different configurations (for now)
- Image Dimensions: 1280x720 pixels.

