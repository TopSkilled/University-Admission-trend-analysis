# Project Plan — University Admission Trend Analysis

## Goal
Build an end-to-end ML system that:
1. Generates realistic synthetic admissions data
2. Trains interpretable ML models
3. Serves an interactive dashboard for analysis, prediction, and segmentation

## Phases

### Phase 1 — Data Generation
**File**: `Code/dataset_generator.py`

- Simulate 1,000 student records with correlated features (CGPA drives Chance_of_Admit)
- Include temporal variation (Year 2018–2024) and program diversity (CS/EE/ME/CE/DS)
- Save as `admissions_data.csv`

### Phase 2 — Model Training
**File**: `Code/model_trainer.py`

- Train Logistic Regression and Random Forest for binary admission decision
- Train Linear Regression for continuous admission probability
- Train KMeans for unsupervised student segmentation
- Serialize all models with `joblib.dump`

### Phase 3 — Dashboard
**File**: `Code/app.py`

- Tab 1 (Overview): KPIs, admission rate, dataset distribution
- Tab 2 (Trend Analysis): Year-over-year GRE/CGPA trends, correlation heatmap
- Tab 3 (ML Models): Accuracy, confusion matrix, feature importance
- Tab 4 (Predictor): User input form → real-time admission probability
- Tab 5 (Clustering): KMeans scatter plot with cluster labels

### Phase 4 — Execution
```bash
pip install -r requirements.txt
python dataset_generator.py
python model_trainer.py
streamlit run app.py
```

## Model Choices Rationale

| Model               | Why chosen                                                          |
|---------------------|---------------------------------------------------------------------|
| Logistic Regression | Interpretable baseline; good for binary classification              |
| Random Forest       | Handles non-linearity; provides feature importances for insights    |
| Linear Regression   | Natural fit for predicting continuous probability score             |
| KMeans              | Simple, visual clustering; segments students without labels         |

## Success Criteria
- All 5 dashboard tabs render without error
- Admission Predictor returns a probability between 0 and 1
- Random Forest accuracy ≥ 85% on test set
- KMeans produces 3 visually distinct clusters on a CGPA vs GRE scatter plot
