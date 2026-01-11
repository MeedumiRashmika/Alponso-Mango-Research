#Alphonso Mango Research Project
Empowering Alphonso Mango Export with AI, Computer Vision & Market Intelligence

This repository contains a multi-component research system focused on improving Alphonso (TJC) mango export quality, packaging, and market intelligence using:

Deep Learning

Computer Vision

Feature Engineering

Time-Series Forecasting

Mobile & Cloud Technologies

The project is divided into four major research components, each developed by a team member.

Research Components Overview
Component	Focus Area
Component 1	Plant Disease Classification (Deep Learning)
Component 2	Mango Quality Grading for Export
Component 3	Intelligent Price Forecasting and Trading System
Component 4	Smart Export Packaging Recommender
Component 1
Plant Disease Classification using Xception (TensorFlow)

This project implements a deep learning–based plant disease detection system using the Xception architecture with transfer learning.

Features include:

Image preprocessing

Data augmentation

Model training and evaluation

Classification metrics

Visualization

Inference

Project Structure
project/
│
├── dataset/                 # Training images (organized by class folders)
│   ├── Class_1/
│   ├── Class_2/
│   └── ...
│
├── models/
│   └── disease-detection.h5 # Trained model (auto-generated)
│
├── plots/
│   └── disease-detection.png # Accuracy / Precision / Recall / AUC plots
│
├── train.py                 # Main training script
└── README.md                # Project documentation

Model Architecture

Base model: Xception (pretrained on ImageNet, frozen layers)

Global Average Pooling

Dense (512 → ReLU)

Dropout (0.5)

Dense (256 → ReLU)

Dropout (0.5)

Dense Softmax Classifier (number of classes)

Requirements
pip install tensorflow numpy pandas pillow scikit-learn matplotlib


TensorFlow 2.x required

Dataset Format
dataset/
│
├── Die Back/
├── Anthracnose/
└── Healthy/

Training
python model.ipynb

Evaluation Output

Accuracy

Precision

Recall

AUC

Classification report

Training plots

Inference Example
result = inference_func("path/to/image.jpg")
print(result)

Component 2
Mango Quality Grading System for Export

AI-based ripeness detection, size measurement, and weight measurement following GAP compliance standards.

This system replaces subjective human inspection with an objective and consistent AI-driven grading approach using images captured from a low-cost setup.

Quality Parameters

Ripeness stage

Physical size

Weight

Final output determines whether a mango is export quality (GAP compliant) or rejected.

Research Motivation

Traditional mango grading for export is:

Time-consuming

Subjective

Inconsistent with GAP standards

This research enables:

Automatic grading

Objective decision-making

Low-cost deployment

Suitability for developing countries

Project Structure
mango-quality-grading/
│
├── dataset/
│   ├── ripe/
│   ├── semi_ripe/
│   └── unripe/
│
├── ripeness_model/
│   └── mango_ripeness_model.py
│
├── size_measurement/
│   └── size_detection.py
│
├── device_setup/
│   └── capture_box_design.md
│
├── main_grading_pipeline.py
└── README.md

How It Works

Step 1 — Image Capture

Smartphone camera at fixed height

White background

Controlled lighting

Fixed mango holder

Reference object (coin or ruler)

Step 2 — Ripeness Detection

Deep learning classification model

Classes: Unripe, Semi-ripe, Ripe

Step 3 — Size Measurement

Pixel-to-centimeter conversion using OpenCV

Compared with GAP size standards

Work Completed

Defined export grading problem

Selected Alphonso (TJC) mango

Captured images under controlled setup

Created labeled dataset

Preprocessed images

Trained ripeness classification model

Component 3
Intelligent Price Forecasting and Trading System

AI-powered local and export market classification with price forecasting.

Predictions

Local markets: Colombo, Dambulla, Pettah

Export markets: Doha, Dubai, London, Paris

Local market prices

Export market prices

Price after 3 months

Uses time-series feature engineering with CatBoost and XGBoost models.

Project Structure
mango-intelligence/
│
├── dataset/
│   └── mango_price_dataset.csv
│
├── train_all_models.py
├── predict_batch_from_json.py
│
├── output/
│   ├── models/
│   └── artifacts/
│
└── README.md

Workflow

Load and clean dataset

Perform feature engineering (27+ features)

Train classifiers and regressors

Save trained models

Predict using JSON input

Model Performance

Local Market Accuracy: 66.25%

Export Market Accuracy: 63.4%

Local Price R²: 0.9989

Export Price R²: 0.9980

3-Month Price R²: 0.9647

Component 4
Smart Export Packaging Recommender System for Alphonso Mangoes

Built using AI/ML, Deep Learning, Feature Engineering, Flutter, and Firebase.

Predicts optimal export packaging using two deep learning regression models:

Shelf Life Prediction

Damage Percentage Prediction

Validates:

Transport duration vs shelf life

Legal and eco-friendly packaging rules

Box layout and quantity

Model 1 — Shelf Life Predictor

Deep Neural Network (Regression)

Dense layers: 256 → 256 → 128 → 64

Optimizer: Adam

Loss: Mean Squared Error (MSE)

Model 2 — Packaging Damage Predictor

Dense layers: 512 → 512 → 256 → 128

Residual connections

Swish activation

L2 regularization

Advanced feature engineering

Team Contribution Checklist
Team Member	Component	Status
Member 1	Plant Disease Classification	Completed
Member 2	Mango Quality Grading	Ripeness Model Completed
Member 3	Intelligent Price Forecasting	Completed
Member 4	Packaging Recommender System	Completed
Overall Research Impact

This research provides an end-to-end intelligent ecosystem for Alphonso mango export, including:

Quality assurance

Disease detection

Market prediction

Smart packaging decision-making

All components are built using low-cost, scalable, and AI-driven solutions.

Git Repository

https://github.com/MeedumiRashmika/Alponso-Mango-Research

## Architecture Diagram



<img width="2019" height="1187" alt="image" src="https://github.com/user-attachments/assets/ae3341c1-4749-416b-b0fe-c70806a360dd" />

## Project Dependencies
- Flutter
- Firebase Firestore
- Python
- TensorFlow / Keras
- NumPy
- Pandas



