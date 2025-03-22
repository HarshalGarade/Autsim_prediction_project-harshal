ASD Classification - Machine Learning Pipeline

Project Overview

This project builds a machine learning pipeline to classify Autism Spectrum Disorder (ASD) using a dataset. The pipeline includes data preprocessing, exploratory data analysis (EDA), handling imbalanced data, model training, hyperparameter tuning, and evaluation.

Table of Contents

Dataset

Installation

Data Preprocessing

Exploratory Data Analysis (EDA)

Handling Imbalanced Data

Model Training & Selection

Evaluation

Model Deployment

Dataset

The dataset is loaded from train.csv.

It contains numerical and categorical features related to ASD screening.

The target variable is Class/ASD (Autism diagnosis: Yes/No).

Installation

Prerequisites

Ensure you have Python installed along with the required libraries.

pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost pickle-mixin

Data Preprocessing

Removed unnecessary columns (ID, age_desc).

Converted age column to integer type.

Fixed country name inconsistencies.

Handled missing values in ethnicity and relation columns.

Performed label encoding for categorical features.

Saved label encoders using pickle for future use.

Exploratory Data Analysis (EDA)

Univariate Analysis

Histograms for age and result.

Box plots to detect outliers.

Count plots for categorical features.

Bivariate Analysis

Correlation heatmap.

Identified class imbalance in the target variable.

Handling Imbalanced Data

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

Model Training & Selection

Used three classifiers:

Decision Tree

Random Forest

XGBoost

Performed 5-fold cross-validation.

Hyperparameter tuning using RandomizedSearchCV.

Selected the best model based on accuracy score.

Saved the best model using pickle.

Evaluation

Used the test dataset to evaluate the best model.

Metrics used:

Accuracy Score

Confusion Matrix

Classification Report

Model Deployment

To deploy the trained model, you can:

Use Flask or FastAPI to create an API.

Deploy on Streamlit for an interactive UI.

Save the model as best_model.pkl and use it for predictions.

