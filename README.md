# Heart Disease Prediction (Classification) 
A machine learning project that predicts whether a patient has heart disease based on a set of medical attributes. Built as a hands-on introduction to classification — covering EDA, preprocessing pipelines, model comparison, and evaluation.

## Overview

Heart disease diagnosis often depends on interpreting a mix of clinical measurements (blood pressure, cholesterol, max heart rate, chest pain type, etc.). This project builds and compares several classification models to predict the presence of heart disease (`0` = No Disease, `1` = Disease) from these attributes.

## Dataset

- **Source:** [Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) (Kaggle, via `kagglehub`)
- Loaded directly in the notebook using the Kaggle Hub API — no manual download needed.

## Workflow

1. **Exploratory Data Analysis (EDA)** — target distribution, feature summaries, missing values, and visual comparisons (age, max heart rate, chest pain type, sex) against the target, plus a correlation heatmap.
2. **Data Preprocessing** — feature/target split, imputation (median/mean for numeric, most frequent for categorical), one-hot encoding of categorical features, and standard scaling, all wrapped in a `scikit-learn` `Pipeline`/`ColumnTransformer` to avoid data leakage.
3. **Model Building** — four classifiers trained and compared:
   - Logistic Regression (baseline)
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
4. **Model Evaluation** — accuracy, precision, recall, F1-score, and confusion matrix analysis for each model, with attention to false negatives (missed diagnoses) as the costliest error type in this context.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- kagglehub

## Repository Structure

```
├── Heart_Disease_Prediction.ipynb   # Main notebook: EDA, preprocessing, modeling, evaluation
└── README.md
```

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook Heart_Disease_Prediction.ipynb
   ```

## Key Insights

- Patients with heart disease tend to have a **lower maximum heart rate**.
- Certain chest pain types (notably asymptomatic) are strongly associated with a higher likelihood of heart disease.
- A correlation heatmap and multi-model comparison help identify which features and algorithms are most predictive.

# Prediction
## Model 	           |   Accuracy	|  Precision (1) | Recall (1)	| F1 (1)|
SVM	                 | SVM	, 0.86 |  SVM,0.85 | SVM ,0.91 | SVM , 0.88|
Random Forest	         0.84	       0.83	           0.89	    0.86
KNN	                  0.84         0.83	           0.89       0.86
Logistic Regression  	0.83	       0.83	           0.87	    0.85

## Author

Jhanvi Khanna

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
