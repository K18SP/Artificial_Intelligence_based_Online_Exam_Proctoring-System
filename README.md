#  AI-Based Online Exam Proctoring System

A full-stack AI-powered exam proctoring platform that monitors students in real-time during online exams. The system combines **computer vision**, **object detection**, **audio monitoring**, and a **Flask web backend** to detect suspicious behaviour and generate detailed integrity reports — all running live through a webcam.

---

##  Key Features

| Feature | Technology | Description |
|---|---|---|
| 👤 Face Detection | Dlib | Detects if student's face is visible at all times |
| 👁️ Eye Gaze Tracking | OpenCV + Dlib | Monitors if student is looking away from screen |
| 👄 Mouth Tracking | Dlib landmarks | Detects talking/whispering during exam |
| 👁️ Blink Detection | Dlib | Unusual blink patterns flagged as suspicious |
| 🗣️ Head Pose Estimation | OpenCV | Detects head turning left/right/up/down |
| 📦 Object Detection | YOLO | Detects prohibited items (phones, books, etc.) |
| 🔊 Audio Detection | PyAudio + Winsound | Flags background voices or unusual sounds |
| 📋 Activity Logging | File I/O | Auto-generates timestamped `activity.txt` report |
| 📊 Integrity Report | Flask + SQL | Full session report saved with violation timestamps |

---

##  Tech Stack

**AI / Computer Vision**
- Python, OpenCV, Dlib
- YOLO (object detection)
- PyAudio, Winsound (audio analysis)

**Backend & Web**
- Flask (server & REST routes)
- SQL Database (session and report storage)
- HTML, CSS, JavaScript (frontend UI)

**Project Structure**
```
Artificial_Intelligence_based_Online_Exam_Proctoring-System/
│
├── ai_engine/              # Core CV models — face, eye, head, blink
├── proctoring/             # Main proctoring logic and monitoring loop
├── speech/                 # Audio detection module
├── backend/                # Flask app logic
├── routes/                 # API route definitions
├── database/               # SQL schema and DB helpers
├── models/
│   └── object_detection_model/  # YOLO weights and config
├── interview/              # Interview mode module
├── static/                 # CSS, JS, images
├── templates/              # HTML templates
├── utils/                  # Shared utility functions
│
├── app.py                  # Main Flask entry point
├── cheating_detector.py    # Core cheating detection logic
├── server.py               # Server configuration
├── main.py                 # Standalone runner
├── activity.txt            # Sample activity log output
├── requirements.txt
└── README.md
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/K18SP/Artificial_Intelligence_based_Online_Exam_Proctoring-System.git
cd Artificial_Intelligence_based_Online_Exam_Proctoring-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Dlib requires CMake. Install it first:
> ```bash
> pip install cmake
> pip install dlib
> ```
> On Windows, you may need Visual Studio Build Tools.

### 3. Run the application
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

---

##  How It Works

```
Student opens exam via browser (Flask web UI)
        ↓
Webcam feed captured → sent to AI Engine
        ↓
Parallel monitoring modules run simultaneously:
  ├── Face Detection (Dlib)
  ├── Eye Gaze + Blink Detection (Dlib landmarks)
  ├── Head Pose Estimation (OpenCV)
  ├── Mouth / Talking Detection (Dlib)
  ├── Object Detection (YOLO — phones, books)
  └── Audio Detection (PyAudio)
        ↓
Violations flagged in real-time → stored in SQL DB
        ↓
Session ends → integrity_report_<timestamp>.txt auto-generated
        ↓
Examiner reviews full activity log post-exam
```

---

##  Sample Output

The system auto-generates timestamped integrity reports like:
```
integrity_report_20260322_002532.txt
integrity_report_20260322_003448.txt
```

Each report contains a timestamped log of all detected violations and behaviours during the exam session.

---

##  Requirements

Key dependencies:
```
flask
opencv-python
dlib
cmake
pyaudio
numpy
torch
ultralytics
sqlalchemy
```

Full list in `requirements.txt`.

---

##  Future Improvements

- [ ] Add browser tab-switch detection
- [ ] Multi-face detection alert (someone else in the room)
- [ ] Cloud deployment (AWS/GCP) for institutional use
- [ ] Admin dashboard with per-student violation summary
- [ ] Integration with LMS platforms (Moodle, Google Classroom)

---

##  Important Notes

- Requires a **webcam** and **microphone** to function
- YOLO model weights must be present in `models/object_detection_model/`
- Tested on **Windows** (Winsound is Windows-only; replace with `playsound` for cross-platform)

---

##  Author

**Kushal Pandya**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/kushal-pandya-302984251/)
[![GitHub](https://img.shields.io/badge/GitHub-K18SP-black)](https://github.com/K18SP)
