# Alponso-Mango-Research
Empowering Alponso Mango Export With Global Marcket Insights


Rashmika D M -IT22583482
🍋 Smart Export Packaging Recommender System for Alphonso Mangoes  
### Using AI/ML + Deep Learning + Feature Engineering + Flutter + Firebase

This project predicts the best export packaging for Alphonso mangoes in Sri Lanka by using  
**two deep learning models**:

- **Model 1 — Shelf Life Prediction**  
- **Model 2 — Damage % Prediction (Packaging Recommender)**  

The system validates:
- Shelf-life vs. transport duration  
- Legal & eco-friendly packaging rules per country  
- Box layout suitability  
- Number of boxes required  
- Final packaging recommendation  

A **Flutter mobile application** allows farmers/exporters to input shipment details and view results.  
All data is stored in **Firebase Firestore**.

---

## 🚀 Features

### ✔ Model 1 – Shelf Life Prediction (Regression)
Predicts shelf life of Alphonso mangoes using:
- Location  
- Ripeness level  
- Mango size  
- Temperature  
- Humidity  

### ✔ Model 2 – Packaging Damage % Prediction (Regression)
Predicts damage % for different box options based on:
- Box dimensions  
- Number of mangoes  
- Material type  
- Temperature shock  
- Handling quality  
- Density  
- Transport duration  
- Shelf life from Model 1  

### ✔ Additional Pipeline Logic
- Compares **shelf life vs. transport duration** (safe or not safe)
- Legal & eco-friendly packaging validation per country
- Suggests **alternative box** if recommended one is not allowed
- Calculates:
  - Mangoes per box
  - Number of boxes required  

---

# 🧠 Machine Learning Overview

The project uses **deep neural networks (DNNs)** built with **TensorFlow/Keras**.

---

## 📌 Model 1 — Shelf Life Predictor

### **Algorithm Used**
- Deep Neural Network (Regression)
- Architecture:  
  - Dense layers: 256 → 256 → 128 → 64  
  - Activation: ReLU  
  - Batch Normalization  
  - Dropout (0.2)

### **Hyperparameters**
| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0008 |
| Loss | MSE |
| Metrics | MAE |
| Batch Size | 32 |
| Epochs | 200 (Early Stopping) |
| Patience | 10 |
| LR Scheduler | ReduceLROnPlateau |


---

## 📌 Model 2 — Packaging Damage Predictor

### **Algorithm Used**
- Deep Neural Network (Regression)
- Feature engineering applied heavily

### **Model Architecture (Final High-Accuracy Version)**  
- Dense layers: 512 → 512 → 256 → 128  
- Residual connection in Dense Block 2  
- Activation: Swish  
- Batch Normalization  
- Dropout (0.3 / 0.2)
- L2 Regularization  

### **Hyperparameters**
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0003 |
| Loss | MSE |
| Metrics | MAE |
| Batch Size | 64 |
| Epochs | 300 |
| Early Stopping | Yes (patience=15) |
| LR Scheduler | ReduceLROnPlateau |



---



