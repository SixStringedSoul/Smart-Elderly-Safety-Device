# Smart-Elderly-Safety-Device
An ESP32-based elderly safety system using machine learning and sensor data for human activity recognition and fall detection.



# Smart Elderly Safety Device – Fall Detection System

## Overview

A machine learning-based elderly safety system designed to
recognize human activities and detect potential falls using
wearable motion sensor data.

The project combines Python-based machine learning,
sensor-data processing and ESP32 hardware to explore
real-time elderly activity monitoring and fall detection.

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- NumPy
- Pandas
- ESP32
- Accelerometer and Gyroscope Data
- Matplotlib

## Datasets

This project uses:

1. SisFall Dataset
2. HAR70+ Dataset

The datasets are used for human activity recognition,
sensor-data preprocessing and fall detection.

Original dataset folder names and structures are preserved.

Raw datasets are not included in this repository.
Please obtain them from their respective official sources
and follow their licensing conditions.

## Machine Learning Models

### 1. Random Forest

- 200 decision trees
- Balanced class weighting
- 42 engineered statistical features
- Validation accuracy: 81.65%

### 2. 1D Convolutional Neural Network

- Implemented using PyTorch
- Uses six sensor channels
- Processes windows of 200 time steps
- Includes convolutional layers, batch normalization,
  ReLU activation, max pooling and dropout.

## Data Processing

The preprocessing pipeline includes:

- Sensor data processing
- Feature extraction
- Window-based segmentation
- Normalization
- Subject-wise train, validation and test splitting

Recorded data split:

| Split | Samples |
| Training | 110,015 |
| Validation | 21,579 |
| Testing | 22,111 |

## Project Structure

The repository contains the source code, model
implementation, evaluation resources and hardware
documentation.

Refer to the individual folders for implementation details.

## Future Improvements

- Improve fall detection performance.
- Optimize model inference for ESP32 deployment.
- Integrate real-time sensor monitoring.
- Develop an emergency notification mechanism.

## Author

Aryan Indu Shanker Misra

B.Tech CSE – Artificial Intelligence and Machine Learning
VIT Chennai
