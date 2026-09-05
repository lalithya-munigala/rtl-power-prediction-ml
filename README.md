# RTL Power Prediction using Machine Learning

A machine learning-based system for predicting RTL (Register Transfer Level) power consumption from RTL design features, with an interactive Tkinter dashboard for training, prediction, analysis, visualization, and model comparison.

## Overview

Traditional RTL power estimation relies heavily on simulation-based methods, which can be time-consuming and computationally expensive.

This project explores machine learning regression models to predict total RTL power consumption using design and activity-related features. The trained models can be saved and reused for future predictions without retraining.

## Features

- RTL dataset analysis
- Data preprocessing and feature transformation
- Multiple machine learning regression models
- Model performance evaluation
- Actual vs. predicted visualization
- Residual analysis
- Error distribution visualization
- Learning curve comparison
- Interactive Tkinter dashboard
- Model saving and loading using `.pkl` files
- Power prediction using trained models

## Input Features

The dataset contains RTL-related features including:

- Benchmark
- Block
- Instruction
- Toggle Count
- Hamming Distance
- Control Activity
- Data Bus Activity
- Stall Cycles
- Static Power (mW)
- Dynamic Power (mW)

The target variable is:

- **Total Power (mW)**

## Data Processing

The preprocessing pipeline includes:

1. Removing unnecessary columns such as `cycle_id` and `total_check`
2. One-hot encoding categorical features
3. Standard scaling of numerical features
4. Splitting the data into 80% training and 20% testing sets
5. Using Scikit-learn's `ColumnTransformer` to maintain consistent preprocessing during training and prediction

## Machine Learning Models

The project implements and compares five regression models:

- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)
- Decision Tree Regressor
- Random Forest Regressor
- Multi-Layer Perceptron (MLP) Regressor

## Model Evaluation

Models are evaluated using:

- R² Score
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

The project also uses:

- Actual vs. Predicted plots
- Residual plots
- Error distribution histograms
- Learning curves

Based on the project evaluation, Random Forest and Neural Network models showed stable validation performance, while the Decision Tree showed faster convergence with signs of overfitting.

## Interactive Dashboard

The Tkinter-based dashboard integrates multiple modules:

- **Prediction** — Generate power predictions using trained models
- **Training** — Train selected regression models
- **Dataset Analysis** — Analyze the input dataset
- **Training Visualization** — Visualize model performance
- **Model Comparison** — Compare learning curves across models

Trained models can be saved and loaded as `.pkl` files so that predictions can be made without retraining.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Tkinter
- Jupyter Notebook

## Project Structure

```text
rtl-power-prediction-ml/
│
├── RTL_GUI.ipynb
└── README.md
