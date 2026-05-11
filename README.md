# Automatic Number Plate Recognition (ANPR) System

AI-powered web-based Automatic Number Plate Recognition system using YOLOv8 and EasyOCR.

This project detects Indian vehicle number plates from images and videos, extracts the plate text using OCR, and displays the results through a professional Flask web interface.

---

# Features

- Indian License Plate Detection using YOLOv8
- OCR-based Character Recognition using EasyOCR
- Image Upload Support
- Video Upload Support
- Duplicate Plate Filtering
- OCR Error Correction
- Indian Number Plate Format Validation
- Dark Themed Web Interface
- Bounding Box Visualization
- Real-Time Detection Pipeline

---

# Tech Stack

## Backend
- Python
- Flask

## Computer Vision & AI
- YOLOv8 (Ultralytics)
- OpenCV
- EasyOCR
- PyTorch

## Frontend
- HTML
- CSS

---

# Project Structure

```bash
MAJOR/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│
├── runs/
│   └── detect/
│       └── train4/
│           └── weights/
│               └── best.pt
│
├── static/
│   ├── style.css
│   └── results/
│
├── templates/
│   └── index.html
│
├── uploads/
│
└── venv/
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd MAJOR
```

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Model Information

This project uses a custom-trained YOLOv8 model trained on an Indian License Plate dataset.

Required model file:

```bash
runs/detect/train4/weights/best.pt
```

---

# Run The Web Application

```bash
python app.py
```

---

# Open In Browser

```text
http://127.0.0.1:5000
```

---

# How It Works

1. Upload Image or Video
2. YOLOv8 detects vehicle number plates
3. Plate region is cropped
4. OCR extracts text
5. Text is cleaned and validated
6. Final detected plates are displayed

---

# Supported Input Formats

## Images
- JPG
- JPEG
- PNG

## Videos
- MP4
- AVI
- MOV

---

# Future Improvements

- Live CCTV Feed Integration
- Database Logging
- Entry/Exit Monitoring
- Real-Time Gate Automation
- Multi-Camera Support
- Admin Dashboard

---

# Deployment

The project can be deployed using:
- Render
- Railway
- Replit
- VPS/Cloud Servers

---

# Authors

Developed as a Major Project for Automatic Vehicle Monitoring and Logging System.