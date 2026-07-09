import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

YOLO_THRESHOLD  = 0.35     
ROI_FRACTION    = 0.6      
GRAYSCALE_LOW   = 180      
CANNY_LOW       = 50
CANNY_HIGH      = 150
HOUGH_THRESHOLD = 80       
HOUGH_MIN_LEN   = 60       
HOUGH_MAX_GAP   = 15       
MIN_PARALLEL    = 2        
ANGLE_TOL       = 12       
STRIPE_SPACING_MIN = 15    
STRIPE_SPACING_MAX = 80    

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = YOLO(MODEL_PATH)
    use_ultralytics = True
    print(f"[INFO] Loaded Ultralytics YOLO model from {MODEL_PATH}")

except Exception as e:
    print("[WARN] Ultralytics failed, fallback to torch model")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    use_ultralytics = False

def classical_zebra_detect(frame, roi_fraction=ROI_FRACTION):
    """
    Returns:
        confirmed (bool) — True if crossing pattern found
        debug_vis (ndarray) — annotated ROI for debugging
        roi_y_start (int) — pixel row where ROI begins
    """
    h, w = frame.shape[:2]
    roi_y_start = int(h * (1 - roi_fraction))
    roi = frame[roi_y_start:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, GRAYSCALE_LOW, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    masked_gray = cv2.bitwise_and(gray, gray, mask=white_mask)
    edges = cv2.Canny(masked_gray, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP
    )

    debug_vis = roi.copy()
    confirmed = False
    parallel_count = 0

    if lines is not None:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-6))
            if angle < ANGLE_TOL:
                horizontal_lines.append((x1, y1, x2, y2))
                cv2.line(debug_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if len(horizontal_lines) >= MIN_PARALLEL:
            y_mids = sorted([int((y1 + y2) / 2) for (_, y1, _, y2) in horizontal_lines])
            
            gaps = [y_mids[i+1] - y_mids[i] for i in range(len(y_mids)-1)]
            valid_gaps = [g for g in gaps if STRIPE_SPACING_MIN <= g <= STRIPE_SPACING_MAX]
            parallel_count = len(valid_gaps) + 1 if valid_gaps else 0
            if parallel_count >= MIN_PARALLEL:
                confirmed = True

    label = f"CV: {'CROSSING DETECTED' if confirmed else 'No pattern'} ({parallel_count} stripes)"
    color = (0, 255, 0) if confirmed else (0, 0, 200)
    cv2.putText(debug_vis, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return confirmed, debug_vis, roi_y_start


def yolo_detect(frame):
    """Returns list of (box [x1,y1,x2,y2], score) above threshold."""
    detections = []
    if use_ultralytics:
        results = model(frame, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                    if conf.item() >= YOLO_THRESHOLD:
                        detections.append((box.cpu().numpy().astype(int), conf.item()))
    else:
        rgb = frame[:, :, ::-1] / 255.0
        t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(t)
        h, w = frame.shape[:2]
        for det in outputs[0]:
            score = det[4].item()
            if score >= YOLO_THRESHOLD:
                box = det[:4].cpu().numpy()
                
                if box[2] <= 1.0:
                    box = box * np.array([w, h, w, h])
                detections.append((box.astype(int), score))
    return detections


def fuse_and_draw(frame, yolo_dets, cv_confirmed, roi_y_start, debug_vis):
    """
    Decision logic:
      • YOLO hit  + CV confirmed  → HIGH confidence (green box + label)
      • YOLO hit  + CV no pattern → MEDIUM — draw box, warn
      • YOLO miss + CV confirmed  → MEDIUM — draw ROI band, warn
      • Both miss                 → no crossing
    """
    result = frame.copy()


    h, w = frame.shape[:2]
    roi_h = debug_vis.shape[0]
    result[roi_y_start:, :] = cv2.addWeighted(
        result[roi_y_start:, :], 0.5, debug_vis, 0.5, 0)

    cv2.line(result, (0, roi_y_start), (w, roi_y_start), (200, 200, 0), 1)

    crossing_detected = False

    for (box, score) in yolo_dets:
        x1, y1, x2, y2 = box
        if cv_confirmed:
            color = (0, 220, 0)
            tag = f"ZEBRA CROSSING {score:.2f} [HIGH]"
            crossing_detected = True
        else:
            color = (0, 165, 255)
            tag = f"Possible crossing {score:.2f} [UNCONFIRMED]"
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        cv2.putText(result, tag, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not yolo_dets and cv_confirmed:
        overlay = result.copy()
        cv2.rectangle(overlay, (0, roi_y_start), (w, h), (0, 200, 100), -1)
        result = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)
        cv2.putText(result, "ZEBRA CROSSING DETECTED (CV only)",
                    (10, roi_y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
        crossing_detected = True

    status_text = "⚠ CROSSING AHEAD — STOP" if crossing_detected else "No crossing detected"
    status_color = (0, 0, 220) if crossing_detected else (180, 180, 180)
    cv2.rectangle(result, (0, 0), (w, 38), (30, 30, 30), -1)
    cv2.putText(result, status_text, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

    return result, crossing_detected

def detect_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    yolo_dets = yolo_detect(frame)
    cv_confirmed, debug_vis, roi_y_start = classical_zebra_detect(frame)
    result, crossing = fuse_and_draw(frame, yolo_dets, cv_confirmed, roi_y_start, debug_vis)

    print(f"[RESULT] YOLO hits: {len(yolo_dets)} | CV confirmed: {cv_confirmed} | Final: {'CROSSING' if crossing else 'CLEAR'}")

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(out_path, result)
    print(f"[INFO] Saved → {out_path}")

    return result, ("CROSSING" if crossing else "CLEAR")


    # cv2.imshow("Zebra Crossing Detection", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def warp_to_topdown(image, corners, output_size=(600,400)):
    """
    Apply homography to warp zebra crossing into top-down view.
    corners: 4 points from zebra crossing contour (ordered: TL, TR, BL, BR)
    """
    src_pts = np.float32(corners)
    dst_pts = np.float32([
        [0,0],
        [output_size[0],0],
        [0,output_size[1]],
        [output_size[0],output_size[1]]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, output_size)
    return warped

def count_zebra_stripes(image_path, model_path="models/zebra.pt", px_to_m=0.025, debug=False):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")
    h, w = image.shape[:2]

    model = YOLO(model_path)
    results = model(image, verbose=False)
    if results[0].masks is None:
        print("No zebra crossing detected")
        return 0, None, 0.0

    zebra_mask = np.zeros((h, w), dtype=np.uint8)
    for m in results[0].masks.data:
        m_np = m.cpu().numpy()
        if m_np.ndim == 3:
            m_np = m_np[0]
        m_bin = (m_np > 0.5).astype(np.uint8)
        m_bin = cv2.resize(m_bin, (w, h))
        zebra_mask = cv2.bitwise_or(zebra_mask, m_bin)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zebra_region = cv2.bitwise_and(gray, gray, mask=zebra_mask)

    _, binary = cv2.threshold(zebra_region, 170, 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25,5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_count = 0
    stripe_heights = []

    for cnt in contours:
        x,y,cw,ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / float(ch+1)
        if aspect_ratio > 2.0 and cw > 150:
            stripe_count += 1
            stripe_heights.append(ch)
            if debug:
                cv2.rectangle(image, (x,y), (x+cw,y+ch), (0,255,0), 2)

    avg_stripe_height_px = np.mean(stripe_heights) if stripe_heights else 0
    crossing_length_px = avg_stripe_height_px * stripe_count
    crossing_length_m = crossing_length_px * px_to_m

    if debug:
        cv2.imwrite("debug_mask_test8.jpg", binary)
        cv2.imwrite("debug_detected_test8.jpg", image)

    return stripe_count, binary, crossing_length_m


if __name__ == "__main__":
    print("=" * 50)
    print("  Zebra Crossing Detector — Pedestrian POV")
    print("  YOLO + Canny + Hough + Grayscale Masking")
    print("=" * 50)
    
    path="images/testing.jpeg"
    count, mask, length_m = count_zebra_stripes(path, "models/zebra.pt", px_to_m=0.025, debug=True)
    print("Final Stripe Count:", count)
    print("Estimated Crossing Length (m):", length_m)
    detect_image(path)
