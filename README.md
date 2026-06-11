# Churn Prediction: CRD vs CRFD Experiment on Random Forest

## 📌 Overview

This project performs a comparative experiment on customer churn prediction using the **MLC Churn dataset**. The goal is to evaluate two experimental designs:

- **CRD(k)** – Completely Randomized Design with `k` repetitions
- **CRFD(max_depth, k)** – Completely Randomized Factorial Design varying `max_depth` and `k`

The model used is **Random Forest**, evaluated using **Stratified K-Fold cross-validation** to handle class imbalance.

---

## 🎯 Objectives

- Compare the effect of purely random repetitions vs. factorial combinations of hyperparameters
- Evaluate model performance using F1-score
- Determine whether tuning `max_depth` significantly impacts churn prediction performance

---

## 📁 Dataset

- **Source**: MLC Churn Dataset (Mobile/Landline Customer Churn)
- **Target variable**: `churn` (binary)
- The dataset path is `data\mlc_churn.csv`
---

## 🧪 Experimental Design

### 1. CRD(k)
- Randomly repeat the Random Forest experiment `k` times
- Each repetition uses the same seed for reproducibility
- `k` ∈ {3, 5, 10}
- Each k is evaluated with Stratified K-Fold with 10 repeats

### 2. CRFD(max_depth, k)
- Factorial combination of:
  - `max_depth` ∈ {3, 5, None}
  - `k` ∈ {3, 5, 10} (repetitions per cell)
- Each combination is evaluated with Stratified K-Fold with 10 repeats

---

## 🛠️ Requirements

Install dependencies:

```bash
pandas
numpy
matplotlib
scikit-learn
statmodels
