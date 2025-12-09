# Telco Customer Churn Prediction using Machine Learning

This project demonstrates how to predict whether a customer will leave a telecommunications company (churn) based on their demographic and account information using various machine learning algorithms. The analysis utilizes the Telco Customer Churn dataset.

## Overview

The goal of this project is to build a classifier that can accurately predict the target variable `Churn` (Yes or No) given features such as customer demographics, services subscribed, contract details, and payment methods.

## Dataset

link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* **The dataset used is the "Telco Customer Churn" dataset. It contains customer information with the following attributes:
* **Demographics:** Gender, SeniorCitizen, Partner, Dependents
* **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
* **Account Information:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
* **Target:** Churn

## Methodology

The project workflow includes the following steps:
1.  **Data Preprocessing**: Converting `TotalCharges` to numeric, handling missing values, and dropping irrelevant columns like `customerID`.
2.  **Feature Engineering**: Handling categorical data using One-Hot Encoding and Label Encoding for the target variable.
3.  **Model Training**: Implementing multiple classification algorithms including Logistic Regression, Random Forest, SVM, and XGBoost.
4.  **Hyperparameter Tuning**: Using GridSearchCV to find optimal parameters for Logistic Regression.
5.  **Evaluation**: Comparing models based on accuracy, precision, recall, F1-score, and ROC-AUC metrics.

## Technologies Used

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* XGBoost

## Models and Performance

The following models were trained and evaluated:

1.  **Logistic Regression**: Achieved approximately 80% accuracy. It demonstrated the best overall performance with the highest ROC-AUC score.
2.  **Random Forest Classifier**: Achieved approximately 79% accuracy.
3.  **XGBoost Classifier**: Achieved approximately 77% accuracy.
4.  **Support Vector Machine (SVM)**: Achieved approximately 73% accuracy.

## How to Run

1.  Clone the repository.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib xgboost
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook predicting-customer-churn-with-machine-learning.ipynb
    ```
