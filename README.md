Iris Classification Project
Overview
This project implements a machine learning pipeline for classifying iris flowers using the famous Iris dataset. The code leverages PyCaret, a low-code machine learning library, to automate the model training and evaluation process.

Project Structure
├── iris.csv              # Input dataset
├── iris_classification.py # Main Python script
├── pipeline.pkl          # Saved model pipeline (generated after execution)
└── README.md             # This file

Requirements
To run this project, you need the following dependencies:

pandas>=1.0.0
pycaret>=2.0.0
scikit-learn>=0.23.0
matplotlib>=3.0.0

Install requirements with:
pip install pandas pycaret scikit-learn matplotlib


**Dataset**
**The Iris dataset contains measurements for 150 iris flowers from three different species:

Setosa

Versicolor

Virginica

Each sample has four features:

Sepal length (cm)

Sepal width (cm)

Petal length (cm)

Petal width (cm)


Code Explanation
Data Loading and Preparation

import pandas as pd

# Load the dataset
data = pd.read_csv('iris.csv')

# Convert to numpy arrays
dataset = data.values
X = dataset[:, 0:4]  # Feature matrix
y = dataset[:, 4]    # Target variable


PyCaret Setup
from pycaret.classification import *

# Initialize PyCaret environment
clf = setup(data=dataset, target=y, normalize=True, normalize_method='zscore')


This configuration:

Sets up a classification experiment

Automatically handles data preprocessing

Applies Z-score normalization to standardize features


Model Comparison and Selection
# Compare multiple models to find the best performer
compare_models()

# Create a Logistic Regression model
lr = create_model('lr')


Evaluation and Visualization
# Plot ROC curve to evaluate model performance
plot_model(lr, plot='auc')


Model Persistence
# Save the entire pipeline for future use
save_model(lr, 'pipeline')


Usage
1. Ensure you have the iris.csv file in your working directory

2. Run the Python script:

python iris_classification.py

3. The script will:

- Load and preprocess the data

- Compare multiple classification algorithms

- Select the best performing model (Logistic Regression in this case)

- Generate performance visualizations

- Save the trained model as pipeline.pkl

Results
The model achieves high accuracy in classifying iris species. The ROC curve visualization demonstrates excellent performance across all three classes.

Model Deployment
The saved pipeline.pkl file contains the entire preprocessing and model pipeline, making it easy to deploy for predictions without retraining.
