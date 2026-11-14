
# GUARDIAN – Smart Elderly Safety and Fall Detection System

### An Intelligent IoT-Based Elderly Monitoring and Fall Detection System

## 1. Project Overview

GUARDIAN is an intelligent elderly safety system designed to recognize human activities and detect potential fall events using wearable motion sensor data.

The project combines Machine Learning, Deep Learning, sensor-data processing and IoT hardware to develop a safety monitoring solution for elderly individuals.

The system uses accelerometer and gyroscope readings to analyze movement patterns, identify abnormal activities and classify potential falls.

It integrates Python-based machine learning pipelines with ESP32 hardware and a web-based application interface.

---

## 2. Project Objectives

- Develop an intelligent fall detection system using motion sensor data.
- Recognize normal human activities and fall events.
- Experiment with multiple Machine Learning and Deep Learning architectures.
- Compare model performance using standard evaluation metrics.
- Optimize classification thresholds to improve detection performance.
- Explore temporal confirmation techniques to reduce false alarms.
- Integrate the software pipeline with ESP32-based hardware.
- Develop a web interface for system interaction and monitoring.

---

## 3. Hardware Components

### Main Hardware

| Component | Purpose |
|---|---|
| ESP32 Development Board | Microcontroller for sensor integration and embedded system development |
| Accelerometer | Measures acceleration along three axes (X, Y, Z) |
| Gyroscope | Measures angular velocity along three axes (X, Y, Z) |
| Wearable Sensor Setup | Collects motion data for activity recognition and fall detection |

### Sensor Data

The machine learning pipeline uses six motion-sensor channels:

- Accelerometer X-axis
- Accelerometer Y-axis
- Accelerometer Z-axis
- Gyroscope X-axis
- Gyroscope Y-axis
- Gyroscope Z-axis

The ESP32 serves as the hardware platform for the elderly safety device.

*Note: The exact IMU sensor module and additional hardware peripherals should be specified after verifying the physical prototype.*

---

## 4. Datasets

The project uses two publicly available human activity and fall detection datasets.

### SisFall Dataset

- Contains recordings of daily activities and simulated falls.
- Used for sensor-data preprocessing and model development.
- Original dataset folder structure is preserved.

### HAR70+ Dataset

- Contains human activity recordings from older adults.
- Used as an additional source of activity recognition data.

### Recorded Dataset Statistics

SisFall dataset inspection:

| Category | Recordings |
|---|---:|
| Total TXT recordings | 4,506 |
| Normal activities (D) | 2,707 |
| Fall recordings (F) | 1,798 |
| SA subjects | 23 |
| SE subjects | 15 |

Raw datasets are not included in this repository. Users must obtain them from their respective sources and comply with their licensing conditions.

---

## 5. Machine Learning and Deep Learning Models

Multiple classification architectures were implemented and evaluated to investigate their suitability for elderly fall detection.

### A. Traditional Machine Learning

**1. Random Forest Classifier**

- Ensemble learning algorithm using multiple decision trees.
- Configured with 200 trees.
- Balanced class weighting.
- Trained using engineered statistical sensor features.
- Recorded validation accuracy: 81.65%.

### B. Deep Learning Models

**2. Artificial Neural Network (ANN)**

- Fully connected neural network architecture.
- Used for classification of processed sensor features.

**3. Convolutional Neural Network (CNN)**

- Applied convolutional operations to sensor time-series data.
- Used to learn local temporal patterns.

**4. CNN-GRU**

- Combines convolutional feature extraction with GRU-based temporal learning.
- Explores spatial/channel-level feature representation and sequential dependencies.

**5. GRU (Gated Recurrent Unit)**

- Recurrent neural network architecture.
- Learns temporal dependencies in sequential sensor readings.

**6. Improved GRU**

- Enhanced GRU-based architecture.
- Evaluated with threshold optimization techniques.

**7. LSTM (Long Short-Term Memory)**

- Recurrent architecture designed to learn longer-term sequential dependencies.
- Evaluated for sensor-based activity classification.

**8. BiGRU (Bidirectional GRU)**

- Processes sequential information in forward and backward directions.
- Evaluated with threshold analysis and temporal confirmation experiments.

---

## 6. Data Preprocessing and Feature Engineering

The preprocessing pipeline includes:

- Dataset inspection and analysis.
- Sensor-data extraction.
- Data cleaning and preprocessing.
- Window-based segmentation.
- Statistical feature extraction.
- Normalization.
- Class balancing and weighting.
- Subject-wise training, validation and testing splits.

### Engineered Features

A total of 42 statistical features were used in the Random Forest pipeline.

Examples include:

- Mean
- Standard deviation
- Minimum
- Maximum
- Other statistical sensor features

### CNN Data Representation

Recorded CNN dataset dimensions:

| Dataset | Samples | Shape |
|---|---:|---|
| Training | 110,015 | 110015 × 200 × 6 |
| Validation | 21,579 | 21579 × 200 × 6 |
| Testing | 22,111 | 22111 × 200 × 6 |

Each sample contains a 200-timestep window with six sensor channels.

---

## 7. Model Evaluation and Performance Analysis

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- False Alarm Rate
- Training and Validation Loss
- Threshold Analysis

### Recorded Random Forest Results

| Metric | Result |
|---|---:|
| Accuracy | 81.65% |
| Precision | 68.78% |
| Recall | 71.70% |
| F1 Score | 70.21% |

### Recorded BiGRU Results

Evaluation threshold: 0.6

| Metric | Result |
|---|---:|
| Accuracy | 82.99% |
| Precision | 76.35% |
| Recall | 61.24% |
| F1 Score | 67.97% |
| False Alarm Rate | 7.93% |

### BiGRU Confusion Matrix

| Classification | Count |
|---|---:|
| True Negative | 14,360 |
| False Positive | 1,236 |
| False Negative | 2,525 |
| True Positive | 3,990 |

These are recorded results from different evaluation pipelines and should not be treated as a direct model ranking without verifying identical test conditions.

---

## 8. Threshold Optimization and Temporal Confirmation

The project also explores techniques beyond basic binary classification.

### Threshold Optimization

- Evaluates different classification probability thresholds.
- Studies the trade-off between precision and recall.
- Analyzes false alarms and missed fall events.

### Temporal Confirmation

- Explores confirmation of fall predictions over time.
- Investigates reducing false alarms caused by isolated predictions.
- Includes evaluation scripts for temporal confirmation rules.

---

## 9. Software Technologies

| Technology | Application |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning model development |
| Scikit-learn | Machine learning and evaluation |
| NumPy | Numerical computations |
| Pandas | Data processing and analysis |
| Matplotlib | Visualization and model evaluation |
| Flask | Web application backend |
| OpenPyXL | Excel result handling |
| ESP32 | Embedded hardware platform |

---

## 10. Project Structure

The repository contains:

- Python scripts for dataset analysis and preprocessing.
- Machine learning and deep learning implementations.
- Model checkpoints.
- Evaluation reports and visualizations.
- Confusion matrices.
- Threshold optimization experiments.
- Temporal confirmation experiments.
- Flask-based web application source code.
- Hardware-related implementation resources.

---

## 11. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-elderly-safety-device.git

cd smart-elderly-safety-device
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

For PyTorch, install the appropriate CPU or CUDA build for your system using the official PyTorch installation instructions.

### Run the Web Application

Navigate to the website directory:

```bash
cd website
```

Install its dependencies if using its separate requirements file:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

---

## 12. Future Enhancements

- Real-time fall detection using wearable sensor streams.
- Improved sensitivity to actual fall events.
- Emergency alert and notification integration.
- Low-latency inference on embedded hardware.
- Improved temporal classification and false alarm reduction.
- Expanded testing with real-world elderly activity data.

---

## 13. Author

**Aryan Indu Shanker Misra**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning  
VIT Chennai

GitHub: https://github.com/SixStringedSoul

---

## Disclaimer

This project is developed for academic and research purposes.

It is not a certified medical device and should not be used as the sole method of emergency or medical monitoring.
