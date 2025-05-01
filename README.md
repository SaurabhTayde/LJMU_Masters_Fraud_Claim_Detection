# Fraudulent Insurance Claim Detection

## Project Overview

This repository contains a case study project focused on building a predictive model to identify fraudulent insurance claims. Developed as an assignment for an **Upgrad course**, this project demonstrates a end-to-end machine learning workflow from data cleaning and exploratory analysis to model building, evaluation, and interpretation.

## Problem Statement

Insurance fraud is a significant financial burden for companies, leading to increased costs and premiums. Traditional manual detection methods are often inefficient and unable to keep pace with the volume of claims. The goal is to leverage historical claim data to automatically assess the likelihood of fraud for incoming claims, thereby allowing for more efficient resource allocation and reducing losses.

## Objective

To develop and evaluate machine learning models capable of accurately classifying insurance claims as either legitimate or fraudulent based on provided historical data, and to derive actionable insights from the chosen model.

## Dataset

The project utilizes a historical dataset containing details of insurance claims and customer information.
*   **Source:** Provided as part of the Upgrad course assignment.
*   **Size:** 1000 claims (rows) with 40 features (columns).
*   **Target Variable:** `fraud_reported` (binary: 'Y' for fraudulent, 'N' for legitimate).
*   **Key Challenge:** The dataset exhibits a significant class imbalance, with legitimate claims being the majority class.

## Methodology

The project followed a structured machine learning methodology:

1.  **Data Loading & Initial Inspection:** Loading the data and performing initial checks for format, size, and basic content.
2.  **Data Cleaning & Preparation:** Handling missing values, addressing illogical data points, and removing redundant or non-informative columns.
3.  **Exploratory Data Analysis (EDA):**
    *   Analyzing distributions and characteristics of individual features (univariate analysis).
    *   Investigating relationships between features and the target variable (bivariate analysis).
    *   Quantifying the class imbalance.
4.  **Data Splitting & Resampling:** Splitting data into training (70%) and validation/test (30%) sets, maintaining class distribution through stratification. Applying **SMOTE** on the training data to address class imbalance.
5.  **Feature Engineering:** Creating new features from existing ones (e.g., date features, claim ratios, interaction terms) to enhance predictive power.
6.  **Feature Transformation:** Encoding categorical variables into numerical format (**One-Hot Encoding**) and scaling numerical features (**StandardScaler**).
7.  **Model Selection & Building:** Exploring **Logistic Regression** and **Random Forest** classifiers.
8.  **Feature Selection:** Using model-specific techniques (**RFECV** for Logistic Regression, **Feature Importance** for Random Forest) to identify the most predictive features.
9.  **Hyperparameter Tuning:** Optimizing the Random Forest model using **Grid Search** with cross-validation to find the best performance parameters.
10. **Model Evaluation:** Evaluating the performance of the final models on the unseen validation/test data using relevant metrics (Accuracy, Balanced Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score), and analyzing performance at different classification thresholds.

## Key Findings & Results

*   The analysis revealed specific features strongly associated with fraud, including `incident_severity`, `insured_hobbies`, `total_claim_amount` and its components, `authorities_contacted`, and `incident_type`.
*   The class imbalance was successfully addressed in the training data using SMOTE, enabling models to learn patterns for the minority fraud class.
*   Feature selection identified key subsets of features for each model, reducing dimensionality while retaining predictive power.
*   Hyperparameter tuning improved the cross-validation performance of the Random Forest model.
*   On the unseen validation/test data (with original imbalance), the **tuned Random Forest model** demonstrated superior performance compared to Logistic Regression, achieving a better balance in identifying fraudulent claims while maintaining high accuracy for legitimate ones.

**Tuned Random Forest Performance on Test Data (using default 0.5 cutoff):**

*   **Accuracy:** 0.8033
*   **Balanced Accuracy:** 0.7377
*   **Specificity:** 0.8673

## Recommendations

Based on the project findings, the tuned Random Forest model is recommended for implementation to prioritize claims for investigation. Leveraging the identified key features can further enhance investigative efficiency and inform proactive risk management strategies in underwriting. Continuous monitoring and retraining of the model are essential for long-term effectiveness.

## Technologies & Libraries Used

*   Python
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   statsmodels
*   imblearn
  

## How to Run

1.  Clone this repository: `git clone https://github.com/SaurabhTayde/LJMU_Masters_Fraud_Claim_Detection`
2.  Navigate to the repository directory.
3.  Ensure you have the necessary libraries installed (`pip install pandas numpy scikit-learn matplotlib seaborn statsmodels imblearn`).
4.  Open and run the Jupyter Notebook to execute the analysis steps.

## Authors:
1. Saurabh Tayde
2. Kasibhatla Shamily Jennymoor



