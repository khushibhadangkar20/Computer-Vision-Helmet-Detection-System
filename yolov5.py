import os
import csv
from datetime import datetime
import time
import winsound

import cv2
import easyocr
import numpy as np
import torch


def iou(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


# Load custom YOLOv5 model with trained weights
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="best.pt",
    force_reload=False,
)

# Model runtime settings
model.conf = 0.65
model.iou = 0.45

# Confidence threshold for filtering detections
CONF_THRESHOLD = 0.5

# EasyOCR reader (English plates)
reader = easyocr.Reader(["en"])

# Ensure violations directory and log file exist
VIOLATIONS_DIR = "violations"
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

LOG_PATH = "violations_log.csv"
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "plate_text", "confidence", "image_path"])

HOURLY_LOG_PATH = "hourly_stats.csv"
if not os.path.exists(HOURLY_LOG_PATH):
    with open(HOURLY_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "hour", "count"])

last_violation_time = 0
COOLDOWN_SECONDS = 10
violation_count = 0


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Unable to read frame from video capture")
        break

    # Run YOLO inference
    results = model(frame)

    # Extract detections: [x1, y1, x2, y2, conf, cls]
    detections = results.xyxy[0].cpu().numpy() if len(results.xyxy) else np.empty((0, 6))

    # Filter by confidence threshold
    detections = detections[detections[:, 4] >= CONF_THRESHOLD] if detections.size else detections

    helmets = []
    heads = []
    plates = []

    # Draw detections and categorize by class
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(conf)
        cls_id = int(cls_id)

        bbox = (x1, y1, x2, y2)
        label_name = results.names.get(cls_id, str(cls_id)) if hasattr(results, "names") else str(cls_id)

        if cls_id == 0:  # Helmet
            helmets.append(bbox)
            color = (0, 255, 0)
        elif cls_id == 1:  # Human head
            heads.append(bbox)
            color = (0, 255, 255)
        elif cls_id == 3:  # Vehicle registration plate
            plates.append({"bbox": bbox, "conf": conf})
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_name} {conf * 100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Determine violations: head without nearby helmet
    violating_heads = []
    for head_box in heads:
        has_helmet = any(iou(head_box, helmet_box) > 0.2 for helmet_box in helmets)
        if not has_helmet:
            violating_heads.append(head_box)

    violation_detected = len(violating_heads) > 0

    # Red border on any active violation
    if violation_detected:
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1] - 1, frame.shape[0] - 1),
            (0, 0, 255),
            8,
        )

    # When violation is detected, capture frame and plate OCR with cooldown
    if violation_detected and plates:
        current_time = time.time()
        if current_time - last_violation_time >= COOLDOWN_SECONDS:
            last_violation_time = current_time

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_filename = f"violation_{timestamp}.jpg"
            image_path = os.path.join(VIOLATIONS_DIR, image_filename)

            # Save full-frame violation image
            cv2.imwrite(image_path, frame)

            # Use the highest-confidence plate for OCR
            best_plate = max(plates, key=lambda p: p["conf"])
            px1, py1, px2, py2 = best_plate["bbox"]

            # Clamp to image bounds
            h, w = frame.shape[:2]
            px1 = max(0, min(px1, w - 1))
            px2 = max(0, min(px2, w))
            py1 = max(0, min(py1, h - 1))
            py2 = max(0, min(py2, h))

            plate_crop = frame[py1:py2, px1:px2]
            plate_text = ""

            if plate_crop.size > 0:
                plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                ocr_results = reader.readtext(plate_rgb, detail=0)
                plate_text = " ".join(ocr_results).strip()

            # Append to CSV log
            human_timestamp = timestamp.replace("_", " ")
            with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [human_timestamp, plate_text, f"{best_plate['conf']:.2f}", image_path]
                )

            # Beep notification
            winsound.Beep(1000, 500)

            # Increment violation counter
            violation_count += 1

            # Update hourly stats
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            hour_str = now.strftime("%H")

            rows = []
            found = False
            if os.path.exists(HOURLY_LOG_PATH):
                with open(HOURLY_LOG_PATH, mode="r", newline="", encoding="utf-8") as f:
                    reader_csv = csv.reader(f)
                    rows = list(reader_csv)

            for row in rows[1:]:
                if row[0] == date_str and row[1] == hour_str:
                    row[2] = str(int(row[2]) + 1)
                    found = True
                    break

            if not found:
                if not rows:
                    rows.append(["date", "hour", "count"])
                rows.append([date_str, hour_str, "1"])

            with open(HOURLY_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    # Draw violation counter on the frame
    cv2.putText(
        frame,
        f"Violations: {violation_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Live Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
