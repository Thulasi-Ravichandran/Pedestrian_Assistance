# 🚶 Pedestrian Assistance System for Visually Impaired Individuals

Pedestrian Safety Assistance Through Traffic Signal and Crosswalk Detection

An intelligent edge-based pedestrian safety assistance system that helps visually impaired individuals safely cross roads by providing real-time voice guidance. The system combines deep learning and classical computer vision techniques to detect zebra crossings, recognize pedestrian traffic lights, estimate crossing distance, and read countdown timers—all without requiring an internet connection.

---

## 📖 Overview

Navigating road crossings is one of the biggest challenges faced by visually impaired individuals. This project provides an offline, real-time assistive solution capable of identifying critical road-crossing information and converting it into spoken instructions.

The system operates entirely on edge devices, making it suitable for portable deployments while maintaining low latency and high accuracy.

---

## ✨ Features

- 🦓 Zebra crossing detection using a fine-tuned YOLOv8n model
- 🚦 Pedestrian traffic light detection and classification
- 🔴🟢 Traffic light color recognition using HSV color segmentation
- 🔢 Countdown timer digit recognition using template matching
- 📏 Crossing distance estimation in meters
- 🎙️ Real-time voice guidance
- 💻 Offline operation (No Internet Required)
- ⚡ Lightweight edge deployment

---

## 🛠 Technologies Used

- Python
- OpenCV
- YOLOv8n (Ultralytics)
- NumPy
- Classical Computer Vision
- HSV Color Segmentation
- Canny Edge Detection
- Hough Line Transform
- Template Matching

---

## 🏗 System Architecture

The system consists of the following modules:

1. **Zebra Crossing Detection**
   - Fine-tuned YOLOv8n model detects zebra crossings.
   - Classical stripe verification using:
     - Canny Edge Detection
     - Hough Line Transform
   - Dual confirmation improves robustness.

2. **Pedestrian Traffic Light Detection**
   - Fine-tuned YOLOv8n model detects pedestrian traffic signals.
   - HSV masking determines:
     - Red
     - Green

3. **Countdown Timer Recognition**
   - Cropped signal regions are color segmented.
   - Digits are recognized using template matching.

4. **Distance Estimation**
   - Stripe geometry is analyzed to estimate crossing length in meters.

5. **Safety Decision Logic**
   - Combines outputs from all modules.
   - Generates clear instructions such as:
     - "Red signal. Please wait."
     - "Green signal. Safe to cross."
     - "Crossing distance is approximately 8 meters."

---

## 📂 Project Structure

```
Pedestrian_Assistance/
│
├── models/
├── templates/
├── images/
├── output/
├── main.py
├── zebra.py
├── traffic_light.py
├── distance_estimation.py
├── countdown.py
├── audio.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Thulasi-Ravichandran/Pedestrian_Assistance.git
```

Navigate to the project

```bash
cd Pedestrian_Assistance
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python main.py
```

---

## 📊 Performance

| Module | Performance |
|---------|------------|
| Zebra Crossing Detection | 92% Accuracy |
| Traffic Light Classification | 95% Accuracy |
| Model Size | 6.3 MB |
| Inference Speed | 2 ms/image |
| Internet Requirement | Not Required |
| GPU Requirement | Not Required |

---

## 🎯 Applications

- Smart mobility assistance
- Assistive technology for visually impaired users
- Edge AI applications
- Intelligent transportation systems
- Smart city accessibility

---

## 🔬 Methodology

The proposed framework combines:

- Deep Learning (YOLOv8n)
- Classical Computer Vision
- HSV Color Analysis
- Template Matching
- Distance Estimation

The outputs from all modules are fused through a Safety Decision Logic module that provides reliable, context-aware guidance for pedestrians.

---

## 📈 Future Improvements

- GPS-assisted navigation
- Dynamic obstacle detection
- Mobile application integration
- Multi-language voice assistance
- Wearable device deployment
- Crosswalk direction estimation

---

## 👩‍💻 Authors

**1. Thulasi Ravichandran**
**2. Dukisha Ganesan**
B.E. Computer Science and Engineering  
Madras Institute of Technology (MIT), Anna University

---

## 📄 License

This project is intended for educational and research purposes.
