# Breast Cancer Diagnosis using K-Nearest Neighbors (Project 2)

## Overview
This project implements a Machine Learning pipeline using K-Nearest Neighbors (KNN) to classify tumors as either Malignant (M) or Benign (B) based on medical features. 

## Features
- Uses `scikit-learn` to implement the K-Nearest Neighbors classification algorithm.
- Data cleaning pipeline that removes unnecessary columns (e.g., IDs, empty Unnamed columns).
- Encoding using `LabelEncoder`.
- Feature scaling using `StandardScaler` to ensure uniform distances.
- Uses `euclidean` distance metric and `k=5` nearest neighbors.
- Outputs detailed evaluation metrics including Accuracy, Confusion Matrix, and a full Classification Report.

## Dataset
This project uses the `KNNAlgorithmDataset.csv`. The target column is `diagnosis`.

## Prerequisites
- Python 3.x
- `pandas`
- `scikit-learn`

Install the required libraries with:
```bash
pip install pandas scikit-learn
```

## How to Run
```bash
python KNN.py
```
