# 🌿 Plant Disease Classification using Xception (TensorFlow)

This project implements a **deep learning–based plant disease detection system** using the **Xception** architecture with transfer learning.  
It supports **image preprocessing, data augmentation, model training, evaluation, classification metrics, visualization, and inference**.

---

## 📁 Project Structure

```
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
```

---

## 🧠 Model Architecture

The system uses **Xception pretrained on ImageNet** as a feature extractor:

- Base model: **Xception (frozen layers)**
- Global Average Pooling
- Dense (512 → ReLU)
- Dropout (0.5)
- Dense (256 → ReLU)
- Dropout (0.5)
- Dense softmax classifier (`num_classes`)

---

## ⚙️ Requirements

Install the required libraries:

```bash
pip install tensorflow numpy pandas pillow scikit-learn matplotlib
```

> TensorFlow 2.x is required.

---

## 📦 Dataset Format

Images must be inside folders named after the class:

```
dataset/
│
├── Die Back/
│   ├── img1.jpg
│   └── img2.jpg
│
├── Anthracnose/
│   ├── img1.jpg
│   └── img2.jpg
│
└── Healthy/
```

---

## 🔧 Training the Model

Run the training script:

```bash
python model.ipynb
```

---

## 📊 Evaluation

The script outputs:

- Categorical accuracy
- Precision
- Recall
- AUC
- Classification report
- Training plots (`plots/disease-detection.png`)

---

## 🔍 Inference

Example:

```python
result = inference_func("path/to/image.jpg")
print(result)
```

---

## 📈 Training Plots

Generated automatically in:

```
plots/disease-detection.png
```

---
