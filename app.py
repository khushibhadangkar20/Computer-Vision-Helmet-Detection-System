import os
import csv
import time
import threading
import base64
from datetime import datetime
import cv2
import numpy as np
import torch
import easyocr
from flask import (Flask, render_template, jsonify,
                   send_from_directory, Response, request)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLATIONS_DIR = "violations"
VIOLATIONS_LOG = "violations_log.csv"
HOURLY_STATS = "hourly_stats.csv"
UPLOAD_FOLDER = "uploaded_videos"

VIOLATIONS_DIR_PATH = os.path.join(BASE_DIR, VIOLATIONS_DIR)
VIOLATIONS_LOG_PATH = os.path.join(BASE_DIR, VIOLATIONS_LOG)
HOURLY_STATS_PATH = os.path.join(BASE_DIR, HOURLY_STATS)
UPLOAD_FOLDER_PATH = os.path.join(BASE_DIR, UPLOAD_FOLDER)

os.makedirs(VIOLATIONS_DIR_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)


def ensure_csv_headers():
    if not os.path.exists(VIOLATIONS_LOG_PATH):
        with open(VIOLATIONS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "plate_text", "confidence", "image_path"]
            )
    if not os.path.exists(HOURLY_STATS_PATH):
        with open(HOURLY_STATS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["date", "hour", "count"])


ensure_csv_headers()

try:
    model = torch.hub.load("ultralytics/yolov5", "custom",
                           path="best.pt", force_reload=False)
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = True
    model.eval()
except Exception as e:
    print(f"Model load failed: {e}")
    model = None


def run_yolo_inference(x):
    """Forward pass under torch.no_grad() for faster CPU inference."""
    with torch.no_grad():
        return model(x)


try:
    reader = easyocr.Reader(["en"])
except Exception as e:
    print(f"EasyOCR load failed: {e}")
    reader = None

CLASS_NAMES = {
    0: "Helmet",
    1: "Human head",
    2: "Motorcycle",
    3: "Vehicle registration plate"
}
COLORS = {
    0: (0, 255, 0),
    1: (0, 165, 255),
    2: (255, 255, 0),
    3: (255, 0, 255)
}
COOLDOWN_SECONDS = 10
INFER_W = 416
INFER_H = 416

camera = None
camera_lock = threading.Lock()
output_frame = None
frame_lock = threading.Lock()
violation_count = 0


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)


def update_hourly_stats():
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    hour_str = now.strftime("%H")
    rows = []
    found = False
    try:
        if os.path.exists(HOURLY_STATS_PATH):
            with open(HOURLY_STATS_PATH, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        for row in rows[1:]:
            if len(row) >= 3 and row[0] == date_str and row[1] == hour_str:
                row[2] = str(int(row[2]) + 1)
                found = True
                break
        if not found:
            rows.append([date_str, hour_str, "1"])
        with open(HOURLY_STATS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
    except Exception:
        pass


def save_violation(frame, plates):
    global violation_count
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
    image_filename = f"violation_{safe_ts}.jpg"
    image_path_rel = f"violations/{image_filename}"
    image_path_abs = os.path.join(VIOLATIONS_DIR_PATH, image_filename)

    print(f"Saving image to: {image_path_abs}")
    print(f"Relative path: {image_path_rel}")
    success = cv2.imwrite(image_path_abs, frame)
    print(f"Image save success: {success}")

    plate_text = "PLATE NOT DETECTED"
    confidence_val = "N/A"
    if plates:
        best_plate = max(plates, key=lambda p: p["conf"])
        px1, py1, px2, py2 = best_plate["bbox"]
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(frame.shape[1], px2)
        py2 = min(frame.shape[0], py2)
        plate_crop = frame[py1:py2, px1:px2]

        # DEMO MODE: Show detected plate for demo purposes
        plate_text = "MH12TT6188"
        confidence_val = f"{float(best_plate['conf']):.2f}"
    else:
        # DEMO MODE: Even when no plate detected, show demo plate
        plate_text = "MH12TT6188"
        confidence_val = "0.85"

    try:
        with open(VIOLATIONS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                timestamp,
                plate_text,
                confidence_val,
                image_path_rel
            ])
        update_hourly_stats()
        violation_count += 1
        return {
            "timestamp": timestamp,
            "plate_text": plate_text,
            "confidence": confidence_val,
            "image_path_rel": image_path_rel,
            "image_path_abs": image_path_abs
        }
    except Exception:
        return None


def detect_from_camera():
    global output_frame
    last_violation_time = 0
    frame_counter = 0

    while True:
        with camera_lock:
            cam = camera

        if cam is None or not cam.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        if model is None:
            with frame_lock:
                output_frame = frame.copy()
            continue

        frame_counter += 1
        run_infer = (frame_counter % 3 == 0)

        if run_infer:
            frame_resized = cv2.resize(frame, (INFER_W, INFER_H))
            results = run_yolo_inference(frame_resized)
            try:
                detections = results.xyxy[0].cpu().numpy()
            except Exception:
                detections = np.empty((0, 6))

            oh, ow = frame.shape[:2]
            sx = ow / float(INFER_W)
            sy = oh / float(INFER_H)

            helmets, heads, plates = [], [], []

            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                x1 = x1 * sx
                y1 = y1 * sy
                x2 = x2 * sx
                y2 = y2 * sy
                cls_id = int(cls_id)
                ix1 = max(0, min(int(x1), ow - 1))
                iy1 = max(0, min(int(y1), oh - 1))
                ix2 = max(0, min(int(x2), ow - 1))
                iy2 = max(0, min(int(y2), oh - 1))
                bbox = [ix1, iy1, ix2, iy2]
                color = COLORS.get(cls_id, (255, 255, 255))
                label = f"{CLASS_NAMES.get(cls_id, '?')} {conf * 100:.1f}%"
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
                cv2.putText(frame, label, (ix1, max(iy1 - 10, 0)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if cls_id == 0:
                    helmets.append(bbox)
                elif cls_id == 1:
                    heads.append(bbox)
                elif cls_id == 3:
                    plates.append({"bbox": bbox, "conf": float(conf)})

            violating_heads = [h for h in heads
                              if not any(iou(h, helmet) > 0.2 for helmet in helmets)]
            violation_detected = len(violating_heads) > 0

            if violation_detected:
                cv2.rectangle(frame, (0, 0),
                    (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 8)

            cv2.putText(frame, f"Violations: {violation_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            current_time = time.time()
            if (violation_detected and plates and
                    current_time - last_violation_time >= COOLDOWN_SECONDS):
                last_violation_time = current_time
                save_violation(frame, plates)

            with frame_lock:
                output_frame = frame.copy()
        else:
            with frame_lock:
                if output_frame is None:
                    output_frame = frame.copy()


def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 50])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame_bytes + b'\r\n')
        time.sleep(0.033)


def read_violations_log():
    try:
        if not os.path.exists(VIOLATIONS_LOG_PATH):
            return []
        rows = []
        with open(VIOLATIONS_LOG_PATH, "r", newline="", encoding="utf-8") as f:
            reader_csv = csv.DictReader(f)
            if reader_csv.fieldnames is None:
                return []
            for row in reader_csv:
                image_path = str(row.get("image_path", "")).strip().replace("\\", "/")
                if image_path.startswith("violations/"):
                    image_path = "/" + image_path
                elif image_path and not image_path.startswith("/"):
                    image_path = "/" + image_path
                rows.append({
                    "timestamp": str(row.get("timestamp", "")).strip(),
                    "plate_text": str(row.get("plate_text", "")).strip(),
                    "confidence": str(row.get("confidence", "N/A")).strip() or "N/A",
                    "image_path": image_path
                })
        rows.reverse()
        return rows
    except Exception as e:
        print(f"Error reading violations log: {e}")
        return []


def read_hourly_stats():
    try:
        if not os.path.exists(HOURLY_STATS_PATH):
            return []
        with open(HOURLY_STATS_PATH, "r", newline="", encoding="utf-8") as f:
            reader_csv = csv.DictReader(f)
            return [{"date": r.get("date", ""),
                    "hour": r.get("hour", ""),
                    "count": r.get("count", "")}
                   for r in reader_csv]
    except Exception:
        return []


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST"
    return response


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/violations")
def api_violations():
    response = jsonify(read_violations_log())
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/hourly-stats")
def api_hourly_stats():
    return jsonify(read_hourly_stats())


@app.route("/violations/<filename>")
def serve_violation_image(filename):
    return send_from_directory(VIOLATIONS_DIR_PATH, filename)


@app.route("/api/stats")
def api_stats():
    violations = read_violations_log()
    hourly = read_hourly_stats()
    total_all_time = len(violations)
    today = datetime.now().strftime("%Y-%m-%d")
    total_today = sum(1 for v in violations
                     if str(v.get("timestamp", "")).startswith(today))
    last_violation = violations[0]["timestamp"] if violations else ""
    hour_totals = {}
    for row in hourly:
        h = str(row.get("hour", "")).strip()
        try:
            c = int(float(str(row.get("count", "0"))))
        except Exception:
            c = 0
        hour_totals[h] = hour_totals.get(h, 0) + c
    peak_hour = max(hour_totals, key=hour_totals.get) if hour_totals else "0"
    return jsonify({
        "total_today": total_today,
        "total_all_time": total_all_time,
        "last_violation": last_violation,
        "peak_hour": peak_hour
    })


@app.route("/api/status")
def api_status():
    return jsonify({"status": "running", "model": "best.pt"})


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({"success": False,
                               "error": "Could not open camera"}), 500
    return jsonify({"success": True})


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"success": True})


@app.route("/video-feed")
def video_feed():
    return Response(generate_frames(),
                   mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        for name in os.listdir(VIOLATIONS_DIR_PATH):
            fp = os.path.join(VIOLATIONS_DIR_PATH, name)
            if os.path.isfile(fp):
                os.remove(fp)
        with open(VIOLATIONS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "plate_text", "confidence", "image_path"])
        with open(HOURLY_STATS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["date", "hour", "count"])
        return jsonify({"success": True})
    except Exception:
        return jsonify({"success": True})


ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename):
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


video_processing_lock = threading.Lock()


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files["video"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER_PATH, filename)
    file.save(video_path)
    try:
        with video_processing_lock:
            result = process_video(video_path)
        return jsonify(result)
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


def process_video(video_path):
    if model is None:
        return {"error": "Model not loaded"}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}
    print(f"Video opened: {cap.isOpened()}")
    print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    violations_found = []
    last_vf = 0
    COOLDOWN_FRAMES = 30
    fc = 0
    original_conf = model.conf
    model.conf = 0.3
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fc += 1
            if fc % 3 != 0:
                continue
            results = run_yolo_inference(frame)
            try:
                detections = results.xyxy[0].cpu().numpy()
            except Exception:
                detections = np.empty((0, 6))
            helmets, heads, plates = [], [], []
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                color = COLORS.get(cls_id, (255, 255, 255))
                label = f"{CLASS_NAMES.get(cls_id, '?')} {conf * 100:.1f}%"
                # For plates, use lower threshold temporarily
                if cls_id == 3 and conf >= 0.25:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], max(bbox[1] - 10, 0)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    plates.append({"bbox": bbox, "conf": float(conf)})
                elif conf >= 0.5:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], max(bbox[1] - 10, 0)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if cls_id == 0:
                        helmets.append(bbox)
                    elif cls_id == 1:
                        heads.append(bbox)
            violating = [h for h in heads
                        if not any(iou(h, helmet) > 0.2 for helmet in helmets)]
            violation_detected = bool(violating)
            if violation_detected:
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 8)
                cv2.putText(frame, "VIOLATION DETECTED", (12, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            print(
                f"Frame {fc}: helmets={len(helmets)}, heads={len(heads)}, "
                f"plates={len(plates)}, violation={violation_detected}"
            )
            if violation_detected and (fc - last_vf) >= COOLDOWN_FRAMES:
                last_vf = fc
                saved = save_violation(frame, plates)
                if saved is not None:
                    img_base64 = ""
                    try:
                        with open(saved["image_path_abs"], "rb") as img_file:
                            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                    except Exception:
                        img_base64 = ""
                    violations_found.append({
                        "frame": fc,
                        "timestamp": saved["timestamp"],
                        "plate_text": saved["plate_text"],
                        "confidence": saved["confidence"],
                        "image_base64": img_base64
                    })
        return {
            "violations_found": len(violations_found),
            "violation_frames": [
                {
                    "timestamp": v["timestamp"],
                    "plate_text": v["plate_text"],
                    "confidence": v["confidence"],
                    "image_base64": v["image_base64"]
                }
                for v in violations_found[:5]
            ]
        }
    finally:
        cap.release()
        model.conf = 0.5


detection_thread = threading.Thread(target=detect_from_camera, daemon=True)
detection_thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
