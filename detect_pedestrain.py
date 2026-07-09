import cv2
import os
from ultralytics import YOLO
from detect_color_p import classify_color
from detect_countdown import detect_countdown

model = YOLO("models/best (4).pt")

image_path = "images/testing.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found")
    exit()

results = model.predict(source=image_path, conf=0.4, device="cpu")

vehicle_count = 0
pedestrian_count = 0
total_count = 0

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        total_count += 1

        class_id = int(cls)
        class_name = r.names[class_id]

        x1, y1, x2, y2 = map(int, box)

        box_color = (255, 255, 255)
        label = class_name

        
        if class_name == "vehicle":
            vehicle_count += 1
            box_color = (255, 255, 0)

        elif class_name == "pedestrain":
            pedestrian_count += 1

            color, _ = classify_color(image, box)

            digits = detect_countdown(
                image,
                box,
                debug=False,
                save_output=True,
                image_name=f"ped_{pedestrian_count}"
            )

            if digits:
                print(f"Pedestrian {pedestrian_count}: {color}, Countdown: {digits}")
                label += f" | {color} | {digits}"
            else:
                print(f"Pedestrian {pedestrian_count}: {color}, No Countdown")
                label += f" | {color}"

            box_color = (0, 255, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

        cv2.putText(image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2)

print("\n===== DETECTION SUMMARY =====")
print("Total Detections:", total_count)
print("Vehicle traffic light Count:", vehicle_count)
print("Pedestrian traffic light Count:", pedestrian_count)

os.makedirs("output", exist_ok=True)
cv2.imwrite("output/final_detection.png", image)

print("\nOutput saved in 'output' folder.")