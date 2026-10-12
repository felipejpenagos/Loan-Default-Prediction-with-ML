<div align="center">

# Loan Default Prediction with Machine Learning

### An end-to-end credit-risk pipeline for an imbalanced vehicle-loan dataset

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](#requirements)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](#requirements)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](#notebooks)

</div>

---

Predicting which borrowers will default on a vehicle loan, using about 233,000 loans and 40 candidate features. Only about 21% of loans default, so the central problem is not fitting a classifier but making one that actually finds the defaults instead of predicting "no default" for everyone.

## Pipeline

1. **Extract, transform, load.** Load and clean the raw loan records; derive borrower age and disbursal month from the date fields.
2. **Exploratory analysis.** Profile categorical and continuous features and their relationship with default.
3. **Feature engineering.** Binning, grouping, scaling, and one-hot encoding.
4. **Model A: logistic regression.** A baseline with a full evaluation suite: confusion matrix, precision, recall, F1, ROC curve.
5. **Model B: random forest.** Tuning the number of trees and maximum depth; capping depth at 15 reduced overfitting and gave the best ROC-AUC (0.65).
6. **Class balancing.** Balanced and manual class weights, up- and down-sampling, and SMOTE, each compared with the same evaluation function.

## What the evaluation shows

The unbalanced logistic regression scores 78% accuracy while catching almost no defaults, which is roughly what predicting "no default" for everyone would score. Rebalancing trades some accuracy for recall: the balanced models identify about half of all defaults. The best trade-off came from a random forest trained on down-sampled data.

## Notebooks

| Notebook | Stage |
|---|---|
| `Extract Transform and Load (ETL) Data CreditRiskML.ipynb` | Loading and cleaning |
| `Exploratory_data_analysis_(EDA)_CreditRiskML.ipynb` | Exploratory analysis |
| `Feature_Engineering_CreditRiskML.ipynb` | Feature engineering |
| `modelA_Logistic_Regression_CreditRiskML.ipynb` | Model A: logistic regression |
| `Advanced_model_evaluation_LogRegression_CreditRiskML.ipynb` | Evaluation metrics and ROC |
| `Random_Forest_classification_CreditRiskML.ipynb` | Model B: random forest and tuning |
| `Accuracy_and_Class_balancing_CreditRiskML.ipynb` | Class weighting, resampling, SMOTE |

`data/` holds the cleaned datasets produced at each stage and a data dictionary.

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## About

Developed as part of the Corporate Finance Institute's machine-learning case study on credit risk.
