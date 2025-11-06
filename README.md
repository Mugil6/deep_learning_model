# 🚀 Deep Learning Project – Automatic Drone Navigation using YOLOv8

## 👥 Group Information
- **Course:** Deep Learning (CSLxxxx)
- **Project Title:** Design a Deep Learning Model for Automatic Drone Navigation
- **Group ID:** 62
- **Members:**
  - [Manisha Bhalla, M25DE1050]
  - [Umang Garg, M25DE1062.]
  - [Sayan Chakraborty, M25DE1044]
  - [Mugilan M, M25DE1022]

---

## 📘 Overview
This project implements an **object detection-based drone navigation system** using **YOLOv8 (Ultralytics)**, built on **PyTorch**.  
The model detects obstacles such as cars, people, and vehicles from aerial videos to support autonomous navigation decisions.

---

## 🧠 Model Description
- **Model Architecture:** YOLOv8n (pre-trained on COCO subset)
- **Framework:** PyTorch
- **Task:** Object Detection for Drone Navigation
- **Training Dataset:** COCO128 (subset of COCO)
- **Training Epochs:** 20  
- **Image Size:** 416×416  
- **Output Classes:** Standard COCO objects (vehicles, persons, etc.)

---

## ⚙️ Folder Descriptions

| Folder/File | Description |
|--------------|-------------|
| `train/train.py` | Training script for YOLO model using COCO128 subset |
| `src/drone_navigation.py` | Main inference + visualization script for object detection |
| `models/best.pt` | Trained YOLOv8 model weights |
| `input/test_2.mp4` | Sample input video |
| `output/` | Folder containing detected output video and plots |
| `requirements.txt` | Python dependencies |
| `instructions.txt` | Dataset download & preprocessing steps |

---

## Installation and Setup

# Create environment
python -m venv venv

# Activate (choose based on OS)
source venv/bin/activate    # Linux / Mac
venv\Scripts\activate       # Windows



# Install dependencies
python -m venv venv

# Train the model
python train/train.py

# Run inference/ testing
python src/drone_navigation.py




