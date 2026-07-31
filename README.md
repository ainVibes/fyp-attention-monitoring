# Intelligent Attention Monitoring System for Smart Classrooms

A real-time, privacy-preserving computer vision system that monitors student attention levels in a classroom setting and automatically alerts lecturers when collective focus drops. Developed as a Final Year Project for the Diploma in Biomedical Electronics Engineering (Laboratory), TVETMARA Ledang.

## Overview

The system uses multi-feature computer vision to estimate attention without relying on facial recognition, prioritizing student privacy while still giving lecturers real-time, actionable feedback on classroom engagement.

Attention is inferred through three signals:
- **Head pose estimation** — detects whether a student is facing the board/camera or looking away
- **Eye Aspect Ratio (EAR)** — detects eye closure and drowsiness
- **Mouth Aspect Ratio (MAR)** — detects yawning
- **Body posture** — detects slouching or disengaged posture

These signals are combined into a per-student attention score, visualized on a live web dashboard.

## Hardware

- Raspberry Pi 5
- Pi Camera Module V3 Wide (120° FOV)

## Tech Stack

- **Computer Vision:** OpenCV, MediaPipe, YOLOv8
- **Backend/Dashboard:** Flask
- **Language:** Python 3.11 (Raspberry Pi, Bookworm)

## Features

- Real-time head pose, eye closure, yawning, and posture detection
- Live Flask dashboard showing:
  - Per-student attention status
  - Attention history graphs
  - Low-attention alerts
- No facial recognition — designed with student privacy as a core constraint
- Optimized frame handling (frame skipping, resolution tuning) to run reliably on Raspberry Pi hardware

## Project Structure

```
├── flask_app.py              # Main application entry point
├── headpose_EAR_MAR_body.py  # Combined multi-feature detection module
├── EAR.py                    # Eye Aspect Ratio detection
├── head_pose.py               # Head pose estimation
├── templates/                 # Flask HTML templates
├── yolov8n.pt                 # YOLOv8 model weights
├── yawn_classifier.tflite     # TFLite yawn classification model
└── dev-log/                   # Early development/learning scripts (day1-day6)
```

## Development Notes

This project went through iterative development, including:
- Migrating camera input from Pi Camera Module (Picamera2) to a USB webcam for reliability
- Diagnosing and resolving a system crash caused by CPU/RAM overload from running YOLOv8, MediaPipe, and Flask concurrently — resolved via frame skipping, reduced resolution, and thermal throttling checks
- Experimenting with a MobileNetV2-based ML classifier for yawning detection (via Google Colab), which plateaued at ~76% accuracy on limited data; the geometric MAR-based approach was retained as the primary method, with the ML experiment documented as a research finding

## Author

**Ain Batrisya Binti Mohd Fadzil**
Diploma in Biomedical Electronics Engineering (Laboratory), TVETMARA Ledang
Supervisor: Muhammad Yusri bin Baharuddin

## License

This project was developed for academic purposes as part of a Final Year Project.
