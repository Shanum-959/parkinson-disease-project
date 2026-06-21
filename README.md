<img width="1440" height="3004" alt="image" src="https://github.com/user-attachments/assets/ab838247-bc8a-4a95-8097-6278275865fa" />
# Parkinson's Disease Detection System

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-green)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A **dual-pathway machine learning web application** that detects Parkinson's Disease using clinical biomarkers and voice acoustic features — built with Flask, Scikit-learn, and trained on two complementary datasets.

---

## Project Overview

| | | |
|---|---|---|
| **93.08%** Clinical model accuracy | **89.74%** Voice model accuracy | **6** ML classifiers trained |
| **2** Datasets used | **12,105** Patient records | **195** Voice recordings |

---

## Model Results — Tabular (Clinical) Dataset

| Model | Accuracy |
|---|---|
| **Gradient Boosting** (Best Model) | **93.08%** |
| Random Forest | 91.82% |
| Decision Tree | 87.42% |
| SVM | 80.19% |
| Logistic Regression | 77.36% |
| KNN | 67.30% |

---

## Datasets

### Clinical Tabular Dataset
| Field | Value |
|---|---|
| Samples | 12,105 |
| Features | 33 |
| Best model | Gradient Boosting |
| Imbalance handling | `class_weight` |
| Source | Kaggle |

### Voice Biomedical Dataset
| Field | Value |
|---|---|
| Samples | 195 |
| Features | 22 |
| Best model | Random Forest |
| Imbalance handling | SMOTE |
| Source | UCI Machine Learning Repository |

---

## Features

- Dual-dataset approach combining clinical biomarkers and voice acoustic features
- Six trained classifiers with full performance comparison (Gradient Boosting, Random Forest, Decision Tree, SVM, Logistic Regression, KNN)
- Proper class imbalance handling using `class_weight` and SMOTE
- Interactive Flask web interface for real-time predictions
- Clean, modern front-end UI

---

## Tech Stack

`Python` · `Flask` · `Scikit-learn` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `Librosa` · `Joblib` · `HTML/CSS`

---

## Run Locally

```bash
# 1 — Clone the repository
git clone https://github.com/Shanum-959/parkinson-disease-project.git
cd parkinson-disease-project

# 2 — Install dependencies
pip install -r requirements.txt

# 3 — Activate virtual environment (Windows)
venv\Scripts\activate

# 4 — Run the app
python app.py

# 5 — Open in browser
http://127.0.0.1:5000
```

---

## Data Preprocessing & Analysis

Before model training, the raw datasets went through a dedicated preprocessing and exploratory analysis pipeline, documented in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb):

- **Outlier removal** using the IQR (Interquartile Range) method
- **Feature scaling** with StandardScaler
- **Dimensionality reduction** using PCA
- **Class imbalance handling** using SMOTE (for the voice dataset) and `class_weight` (for the clinical dataset)
- **Exploratory data analysis** with visualizations built using Matplotlib and Seaborn

This preprocessing and analysis stage was applied before feeding data into the six classifiers compared above.

---

## Project Structure

```
parkinson-disease-project/
├── notebooks/
│   └── analysis.ipynb   # Data preprocessing & exploratory analysis (IQR, PCA, SMOTE, scaling, visualizations)
├── models/          # Trained ML models
├── templates/        # HTML templates
├── app.py             # Flask application entry point
├── requirements.txt   # Project dependencies
└── README.md
```

---

## Contact

Developed by **Shanum Shahzad**
Email: shanumshahzad01@gmail.com
[GitHub](https://github.com/Shanum-959) · [LinkedIn](https://www.linkedin.com/in/shanum-shahzad-a6b130296/)
