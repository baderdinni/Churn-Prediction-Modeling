# Annuity Accumulation Products & Churn Behavior Analysis

## Overview
This project analyzes the churn behavior of annuity accumulation products, which are financial instruments that provide guaranteed returns for retirement savings. The analysis focuses on churn rates at different stages of the annuity maturity cycle, as illustrated in the provided image.

## Problem Statement
- GAFG has a rich dataset of surrender/churn activity on annuity products, but predictions are being made using judgment-based approaches rather than data.
- Existing models rely on complex tables of churn rates that perform poorly across products and are difficult to maintain.
- Several important factors were not included in the current model.
- The data is a collection of multiple disparate datasets and business blocks, each with a different structure.

## Approach
- Gathered, normalized, joined, and validated data from multiple databases to answer fundamental questions about surrender behavior.
- Examined the impact of various factors such as contract renewal rates, contract structures, distribution firms, and demographic/policyholder features on churn rates.
- Used a tree-based bagging approach (Random Forest) and a Feed Forward Neural Network to model complex features that predict surrenders.

## Results
- Replaced the legacy judgment-based model, which relied on complex tables of surrender rates, with a unified data-driven model.
- Extrapolated surrender behavior in retail products with limited data, using insights from reinsured products, leading to better data-driven decisions.
- Developed a predictive model to quantify the influence of various factors on surrender with greater accuracy and explanatory power.
- Proposed strategies to improve pricing, sales, and product development investment decisions.

## Features
- **Data Processing**: Extracts and processes annuity-related churn data.
- **Churn Rate Analysis**: Evaluates customer churn behavior at different stages (before renewal, at renewal, and later years).
- **Machine Learning Models**: Implements Random Forest and Neural Network models for churn prediction.
- **Visualization**: Generates charts to represent churn behavior patterns.
