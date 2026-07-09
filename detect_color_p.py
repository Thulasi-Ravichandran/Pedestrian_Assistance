import cv2
import numpy as np
def classify_color(image, box):
    """
    Takes full image and bounding box coordinates
    Returns detected color (RED or GREEN)
    """
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
    
    if crop.size == 0:
     return "UNKNOWN", (255, 255, 255)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    #RED MASK 
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2
    # GREEN MASK
    lower_green = np.array([35, 70, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    # COUNT PIXELS
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    if red_pixels > green_pixels:
        return "RED", (0, 0, 255)
    else:
        return "GREEN", (0, 255, 0)
    