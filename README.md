# Loan Default Prediction with Machine Learning

This project is an end-to-end case study focused on predicting loan defaults using classification models. It includes the complete **machine learning pipeline**: data loading and cleaning, exploratory data analysis, feature engineering, model training, evaluation, and hyperparameter tuning. Both **logistic regression** and **random forest classifiers** are implemented and compared.

A key challenge addressed in the project is class imbalance– defaults represent only ~21% of the dataset. Several strategies were applied to handle this, including class weighting (`balanced` and manual), resampling (upsampling/downsampling), and synthetic oversampling (**SMOTE**). Model performance was evaluated using metrics such as **accuracy**, **precision**, **recall**, **F1 score**, **ROC-AUC**, and **confusion matrices**. The best-performing model was a downsampled **random forest**, which provided improved recall while maintaining reasonable precision and interpretability.

## Structure

> **"Extract Transform and Load (ETL) Data NB"**  
> *Load and preprocess the data.*

> **"Exploratory Data Analysis (EDA) NB"**  
> *Exploratory data analysis.*

> **"Feature Engineering NB"**  
> *Feature transformations, scaling, and one-hot encoding.*

> **"Model I Logistic Regression NB"**  
> *Logistic regression model.*

> **"Advanced Model Evaluation Log Regression"**  
> *Custom model evaluation metrics.*

> **"Model II Random Forest Classification NB"**  
> *Random forest with hyperparameter tuning.*

> **"Accuracy and Class Balancing NB"**  
> *Handling class imbalance using weights, resampling, and `SMOTE`.*




## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

## Outcome
By iterating through multiple balancing and modeling techniques, the study demonstrates how thoughtful preprocessing and evaluation can significantly enhance classification performance in imbalanced datasets.
