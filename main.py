import cv2
from detect_color_p import classify_color
from detect_countdown import detect_countdown
from zebra import detect_image, count_zebra_stripes
from detect_pedestrain import model  # reuse YOLO model if needed

WALKING_SPEED = 1.0

def safety_decision(signal_color, crossing_status, countdown, crossing_length):
    if crossing_status == "CLEAR":
        return "NO CROSSING DETECTED"
    if crossing_status == "CROSSING" and signal_color == "RED":
        return "DO NOT CROSS"
    if crossing_status == "CROSSING" and signal_color == "GREEN" and countdown is None:
        return "SAFE TO CROSS"
    if crossing_status == "CROSSING" and signal_color == "GREEN" and countdown is not None and crossing_length > 0.0:
        required_time = crossing_length / WALKING_SPEED
        if countdown > required_time:
            return "SAFE TO CROSS"
        else:
            return "WAIT FOR NEXT SIGNAL"
    return "NO CROSSING DETECTED"


if __name__ == "__main__":

    path = "images/testing.jpeg"
    image = cv2.imread(path)
    
    stripe_count, mask, crossing_length = count_zebra_stripes(path, "models/zebra.pt", px_to_m=0.025, debug=True)
    _, crossing_status = detect_image(path)   

    print("Final Stripe Count:", stripe_count)
    print("Estimated Crossing Length (m):", crossing_length)

    results = model.predict(source=path, conf=0.25, device="cpu")
    signal_color, countdown = None, None
    for r in results:
        for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            class_name = r.names[int(cls)]
            if class_name == "pedestrain":
                signal_color, _ = classify_color(image, box)

                countdown = detect_countdown(image, box, debug=False)

    decision = safety_decision(signal_color, crossing_status, countdown, crossing_length)
    print("[FINAL DECISION]:", decision)

    cv2.rectangle(image, (0, 0), (image.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(image, decision, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite("output/testing.jpeg", image)
    print("[INFO] Saved → output/final_output.png")

