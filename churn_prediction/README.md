Here’s a comprehensive `README.md`:

---

# Churn Prediction for Annuity Products

This repository contains code for predicting churn (surrender) for annuity products using **Random Forest** and **Feed-Forward Neural Network (FFNN)** models. The project includes data preprocessing, model training, cross-validation, evaluation, and visualization of results.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)

---

## Project Overview

The goal of this project is to predict customer churn for annuity products using machine learning models. The dataset includes features such as policy details, customer demographics, and financial metrics. Two models are implemented:
1. **Random Forest Classifier**: A tree-based ensemble model.
2. **Feed-Forward Neural Network**: A deep learning model with two hidden layers.

The project also includes:
- Data preprocessing (encoding, scaling, and splitting).
- Cross-validation for model evaluation.
- Visualization of training and validation metrics (loss and AUC).
- Time-series analysis of predicted vs. actual churn rates.

---

## Repository Structure

```
churn-prediction/
├── data/                     # Folder containing the dataset
│   └── churn_data.csv        # Raw dataset file
├── models/                   # Folder containing model implementations
│   ├── __init__.py
│   ├── random_forest.py      # Random Forest model class
│   └── feed_forward_nn.py    # Feed-Forward Neural Network model class
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── backtesting_analysis.py # Functions for backtesting the model results
│   └── data_preprocessing.py # Functions for data preprocessing
│   └── feature_importance.py # Functions for feature importance from the model
│   └── visualization.py      # Functions for plotting results
├── scripts/                  # Main script to run the project
│   └── train.py              # Script to train and evaluate models
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/churn-prediction.git
   cd churn-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Place your dataset (`churn_data.csv`) in the `data/` folder.

2. Run the main script to train the models and generate results:
   ```bash
   python scripts/train.py
   ```

3. The script will:
   - Preprocess the data.
   - Train and evaluate the Random Forest and Neural Network models.
   - Generate plots for cross-validation results, training/validation loss, AUC, and monthly churn analysis.

---