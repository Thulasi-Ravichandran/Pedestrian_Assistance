import cv2
import numpy as np
import os

TEMPLATES = {}

def load_templates(path="templates"):
    for i in range(10):
        img = cv2.imread(f"{path}/{i}.png", 0)
        if img is not None:
            img = cv2.resize(img, (28, 28))
            TEMPLATES[i] = img

load_templates()

def match_template(digit_img):
    best_score = -1
    best_digit = None

    for digit, template in TEMPLATES.items():
        res = cv2.matchTemplate(digit_img, template, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]

        if score > best_score:
            best_score = score
            best_digit = digit

    if best_score > 0.4:
        return best_digit
    return None


def detect_countdown(image, box, debug=False, save_output=True, image_name="img"):

    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    roi = cv2.resize(roi, None, fx=2, fy=2)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = np.array([35, 70, 50])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    digit_mask = red_mask + green_mask

    kernel = np.ones((3, 3), np.uint8)
    digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_CLOSE, kernel)
    digit_mask = cv2.dilate(digit_mask, kernel, iterations=1)

    if save_output:
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(f"output/mask_{image_name}.png", digit_mask)

    contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 20 and w > 10:
            digit_regions.append((x, y, w, h))

    if len(digit_regions) == 0:
        return None

    digit_regions = sorted(digit_regions, key=lambda x: x[0])

    detected_digits = ""

    for idx, (x, y, w, h) in enumerate(digit_regions):
        digit_img = digit_mask[y:y+h, x:x+w]
        digit_img = cv2.resize(digit_img, (28, 28))

        digit = match_template(digit_img)

        if digit is not None:
            detected_digits += str(digit)

        if save_output:
            cv2.imwrite(f"output/digit_{image_name}_{idx}.png", digit_img)

    return detected_digits if detected_digits != "" else None