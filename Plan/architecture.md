# System Architecture — University Admission Trend Analysis

## Overview

This system ingests a synthetic graduate admissions dataset, trains multiple ML models, and serves an interactive Streamlit dashboard for trend analysis, predictions, and clustering.

## Data Flow

```
dataset_generator.py
        │
        ▼
admissions_data.csv  (1,000 rows × 11 columns)
        │
        ├──▶ model_trainer.py
        │           │
        │           ├── logistic_regression.pkl
        │           ├── random_forest.pkl
        │           ├── linear_regression.pkl
        │           └── kmeans.pkl
        │
        └──▶ app.py (Streamlit Dashboard)
                    │
                    ├── Tab 1: Overview
                    ├── Tab 2: Trend Analysis
                    ├── Tab 3: ML Models
                    ├── Tab 4: Admission Predictor
                    └── Tab 5: Clustering
```

## Components

### 1. Dataset Generator (`dataset_generator.py`)
- Generates 1,000 synthetic records modeled after the Kaggle UCLA Graduate Admissions dataset
- Seeds NumPy random state for reproducibility
- Outputs `admissions_data.csv` in the same directory

### 2. Model Trainer (`model_trainer.py`)
- Loads `admissions_data.csv`
- Splits data 80/20 train/test
- Trains and persists four models:
  - **Logistic Regression** — binary classifier for `Admitted`
  - **Random Forest Classifier** — classifier with feature importances
  - **Linear Regression** — regressor for `Chance_of_Admit`
  - **KMeans (k=3)** — unsupervised clustering on standardized features
- Prints evaluation metrics to console

### 3. Streamlit Dashboard (`app.py`)
- Single-file Streamlit application; loads CSV and `.pkl` models at startup
- Five tabs, all driven by Plotly for interactive charts

## Dataset Schema

| Column           | Type  | Range / Values         |
|------------------|-------|------------------------|
| GRE_Score        | int   | 290–340                |
| TOEFL_Score      | int   | 92–120                 |
| University_Rating| int   | 1–5                    |
| SOP              | float | 1.0–5.0 (0.5 steps)    |
| LOR              | float | 1.0–5.0 (0.5 steps)    |
| CGPA             | float | 6.0–10.0               |
| Research         | int   | 0 or 1                 |
| Chance_of_Admit  | float | 0.34–0.97              |
| Admitted         | int   | 0 or 1 (threshold 0.65)|
| Year             | int   | 2018–2024              |
| Stream           | str   | CS, EE, ME, CE, DS     |

## ML Models

| Model                | Library           | Target           | Purpose                        |
|----------------------|-------------------|------------------|--------------------------------|
| Logistic Regression  | sklearn           | Admitted (0/1)   | Binary classification          |
| Random Forest        | sklearn           | Admitted (0/1)   | Feature importance + accuracy  |
| Linear Regression    | sklearn           | Chance_of_Admit  | Probability estimation         |
| KMeans (k=3)         | sklearn           | Unsupervised     | Student profile segmentation   |

## Technology Stack

| Layer        | Technology              |
|--------------|-------------------------|
| Language     | Python 3.x              |
| ML           | scikit-learn            |
| Visualization| Plotly Express + Graph Objects |
| Dashboard    | Streamlit               |
| Serialization| joblib (via sklearn)    |
| Data         | pandas, NumPy           |

## Deployment

```
streamlit run app.py
```
Serves at `http://localhost:8501` — open in Chrome.
