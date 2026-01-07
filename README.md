# Soil Type Prediction and Crop Recommendation Using Hybrid AI

## Overview
This project presents a **hybrid artificial intelligence framework** for predicting soil types and recommending suitable crops by combining **image-based deep learning** and **tabular-data machine learning** techniques. The system integrates visual soil texture analysis with numerical soil and environmental parameters to provide reliable, data-driven agricultural recommendations.

The solution is designed for **precision agriculture** and demonstrates how multi-modal learning can improve decision-making in real-world farming scenarios.

---

## Problem Statement
Traditional crop selection relies heavily on manual soil testing and farmer experience, which can be time-consuming and inconsistent. With increasing climate variability and food demand, there is a need for **automated, intelligent systems** that can analyze soil conditions accurately and recommend optimal crops.

---

## Key Contributions
- Designed a **dual-branch hybrid model** combining CNN-based soil image classification and ensemble-based crop recommendation.
- Performed extensive **model comparison and optimization** using transfer learning and ensemble methods.
- Demonstrated the impact of **SMOTE-based data balancing**, improving crop recommendation accuracy by over **30%**.
- Built a scalable architecture suitable for **future IoT and smart-farming integration**.

---

## System Architecture
The system consists of two main modules:

### 1. Soil Type Prediction (Image-Based)
- Classifies soil images into:
  - Alluvial
  - Black
  - Clay
  - Red

### 2. Crop Recommendation (CSV-Based)
- Recommends crops using soil nutrients and environmental parameters:
  - Nitrogen (N), Phosphorus (P), Potassium (K)
  - pH
  - Temperature, Humidity, Rainfall

The outputs from both modules are fused to ensure crop recommendations align with detected soil types.

---

## Dataset
This project uses **two datasets**.

### 1. Tabular Dataset (Crop Recommendation)
- Soil fertility and environmental parameters with crop labels
- Initially imbalanced with **3,867 samples**
- Balanced using **SMOTE**, expanded to **15,120 samples**

**Dataset Source (Public):**  
https://www.kaggle.com/code/tmleyncodes/research-work-agrosense-agricultural-ai

### 2. Image Dataset (Soil Type Classification)
- Approximately **3,600 soil images**
- **4 soil classes:** Alluvial, Black, Clay, Red
- **Train/Test split:** 80% / 20%
- **Image size:** 224 × 224 (RGB)

> **Note:**  
> Datasets are not included in this repository due to size constraints. All datasets are publicly available and linked above for reproducibility.

---

## Methodology

### Data Preprocessing

**CSV Dataset**
- Data cleaning and label encoding
- Feature scaling
- SMOTE oversampling for class balance
- 80/20 train-test split

**Image Dataset**
- Image resizing and normalization
- Data augmentation (rotation, zoom, horizontal flip)
- Label encoding
- Balanced train-test split

---

## Models Used

### Crop Recommendation (Tabular Data)
- Random Forest
- XGBoost
- CatBoost
- AdaBoost
- CNN + Random Forest (Hybrid)

**Best Model:**  
- Random Forest → **85.42% accuracy (after SMOTE)**

---

### Soil Type Prediction (Image Data)
- Custom CNN
- Transfer Learning Models:
  - ResNet50
  - InceptionV3
  - Xception

**Best Model:**  
- Xception (Adam + ReLU) → **79.03% accuracy**
- Improved generalization using custom **L3 regularization**

---

## Results Summary

### Crop Recommendation Performance
| Model | Accuracy |
|------|----------|
| Random Forest | 85.42% |
| CNN + Random Forest | 84.76% |
| XGBoost | 83.53% |
| CatBoost | 83.99% |

SMOTE improved accuracy by approximately **34%**.



---

### Soil Image Classification Performance
| Model | Accuracy |
|------|----------|
| Xception | 79.03% |
| InceptionV3 | 77.30% |
| ResNet50 | 57.90% |

---

## Tools & Technologies
- **Programming:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn
- **Deep Learning:** TensorFlow, Keras
- **Models:** Random Forest, XGBoost, CatBoost, CNN, Xception
- **Data Processing:** SMOTE, Feature Scaling
- **Visualization:** Matplotlib, Seaborn

---

## Reproducibility
1. Clone the repository  
2. Install dependencies  
3. Download datasets from the links provided  
4. Run the Jupyter notebooks in sequence  

---

## Future Work
- Integration with **IoT-based real-time soil sensors**
- Adoption of **Vision Transformers (ViT)** and EfficientNet variants
- Explainable AI (SHAP / LIME) for model transparency
- Multi-modal neural networks for deeper data fusion
- Crop yield prediction and disease detection
- Federated learning for multi-region scalability
