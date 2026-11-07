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

## Project Structure

```
rtl_power/pipeline.py   # GUI-free core: data loading, preprocessing, models, metrics, schema helpers
tests/test_pipeline.py  # pytest unit tests
RTL_GUI.ipynb           # Tkinter dashboard
```

## Running the Tests

```bash
pip install -r requirements.txt
pytest --cov=rtl_power
```

The suite has 26 tests with 100% line coverage of `rtl_power/pipeline.py`. It covers feature selection, preprocessing (one-hot encoding, scaling, unseen categories), training and prediction for all five models, reproducibility, saving and loading models, metric calculations and the GUI schema helpers. GitHub Actions runs it on every push.

## Note on Target Leakage

In this dataset `total_power_mW` is exactly `static_power_mW + dynamic_power_mW`, so using both components as inputs lets a model add them up instead of learning from RTL activity. The pipeline drops these two columns by default (`exclude_leakage=True`), and a test checks this property of the data. Even without them the models still score well (30k-row sample, 80/20 split):

| Model | R² with leakage | R² without leakage |
|---|---|---|
| Random Forest | 1.0000 | 0.9966 |
| Decision Tree | 0.9999 | 0.9934 |
| KNN | 0.9935 | 0.9906 |
