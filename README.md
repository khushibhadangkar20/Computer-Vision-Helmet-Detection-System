# 🪖 Smart Helmet Violation Detection System

### AI-Based Campus Safety Enforcement | Computer Vision Project

![Python](https://img.shields.io/badge/Python-3.14-blue)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Custom%20Trained-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Number%20Plate-orange)

---

## 📌 Problem Statement

College campuses face challenges enforcing helmet rules at entry gates. Manual checking is inefficient, inconsistent, and requires dedicated staff. This system automates the entire process using AI and computer vision.

---

## 💡 Solution

A real-time AI system placed at the college entry gate that:

* Detects riders without helmets automatically
* Captures violation images with timestamp
* Reads number plate using OCR
* Logs all violations for admin review

---

## 🎯 Features

| Feature                          | Status |
| -------------------------------- | ------ |
| Real-time helmet detection       | ✅ Done |
| No-helmet violation trigger      | ✅ Done |
| Violation image capture          | ✅ Done |
| Number plate detection           | ✅ Done |
| OCR plate text reading           | ✅ Done |
| 10 second cooldown timer         | ✅ Done |
| Live violation counter on screen | ✅ Done |
| Red border alert on violation    | ✅ Done |
| Confidence score logging         | ✅ Done |
| Hourly violation stats           | ✅ Done |
| Web dashboard                    | ✅ Done |
| Video upload analysis            | ✅ Done |
| Gallery view                     | ✅ Done |

---

## 🧠 AI Model

* **Architecture:** YOLOv5s (Custom Trained)
* **Confidence Threshold:** 0.25
* **NMS IoU Threshold:** 0.45

### Classes Detected:

* `0 → Helmet`
* `1 → Human Head`
* `2 → Motorcycle`
* `3 → Vehicle Registration Plate`

---

## 🛠️ Tech Stack

| Category        | Technology          |
| --------------- | ------------------- |
| AI Model        | YOLOv5              |
| Language        | Python 3.14         |
| Computer Vision | OpenCV              |
| OCR             | EasyOCR             |
| Deep Learning   | PyTorch             |
| Web Dashboard   | Flask + HTML/CSS/JS |

---

## 📁 Project Structure

```
helmet-violation-detection/
│
├── app.py                 # Main Flask web application
├── best.pt                # Trained YOLOv5 model weights
├── requirements.txt       # Python dependencies
│
├── violations/            # Captured violation images
│   ├── demo_violation_*.png
│   └── violation_*.jpg
│
├── violations_log.csv     # Violation log
│   └── timestamp, plate_text, confidence, image_path
│
├── hourly_stats.csv       # Hourly violation statistics
│   └── date, hour, count
│
├── templates/
│   └── index.html         # Web dashboard UI
│
└── uploaded_videos/       # Uploaded test videos
```

---

## ⚙️ How It Works

```
Camera/Video Feed
↓
Frame Extraction + Resize to 416x416
↓
YOLOv5 Inference → Get Bounding Boxes
↓
For each detection:

Class 0 (Helmet) → helmets[]
Class 1 (Head) → heads[]
Class 3 (Plate) → plates[]

↓
Violation Check: head without helmet (IoU < 0.2)
↓
├── NO VIOLATION → Continue monitoring
└── VIOLATION DETECTED
↓
Red border + Counter increment
↓
Capture frame → Save to /violations
↓
Log to violations_log.csv + hourly_stats.csv
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/Tanmayy-k/helmet-violation-detection
cd helmet-violation-detection

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py

# Open browser
http://localhost:5000
```

---

## 📊 API Endpoints

| Endpoint            | Method | Description                     |
| ------------------- | ------ | ------------------------------- |
| `/`                 | GET    | Main dashboard                  |
| `/api/violations`   | GET    | All violation records           |
| `/api/stats`        | GET    | Statistics (today, total, peak) |
| `/api/camera/start` | POST   | Start camera                    |
| `/api/camera/stop`  | POST   | Stop camera                     |
| `/api/upload-video` | POST   | Upload & analyze video          |
| `/api/clear`        | POST   | Clear all violations            |
| `/video-feed`       | GET    | MJPEG video stream              |

---

## 📝 Sample Data

| Timestamp           | Plate      | Confidence | Image                             |
| ------------------- | ---------- | ---------- | --------------------------------- |
| 2026-04-27 09:15:00 | MH12XX3456 | 0.88       | demo_violation_1.png              |
| 2026-04-27 18:54:57 | MH12TT6188 | 0.85       | violation_2026-04-27_18-54-57.jpg |

---

## 🏫 Use Case

Deployed at college entry gate to enforce helmet rules, reduce manual checking, and improve campus safety.

### Future Extensions:

* Traffic enforcement systems
* CCTV integration
* Automated fine generation

---

## 👥 Team

* Tanmay Kshirsagar
* Atharva Borate
* Khushi Bhadangkar
* Anushri Ghosh

---

## 📄 License

MIT License
