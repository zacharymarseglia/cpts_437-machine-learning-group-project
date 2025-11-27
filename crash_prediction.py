"""
Tacoma Crash Prediction - Risk Scoring
- Logistic Regression + Decision Tree Ensemble
- Risk scoring only (no classification)

Author: Kendall Reid
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from models import LogReg, DecisionTree

# Load data
data = pd.read_csv("cleaned_tacoma_crashes_with_weather.csv")

# Relevant features
feature_cols = [
    'hour', 'day_of_week', 'is_weekend', 'month',
    'time_of_day', 'road_type', 'road_surface',
    'PRCP', 'SNOW', 'TAVG', 'weather_category',
    'alcohol_related', 'is_freezing'
]

# One-hot encode categorical features
X_pos = pd.get_dummies(data[feature_cols], drop_first=True)
y_pos = np.ones(len(X_pos))

# Allows alignment for new columns
X_columns = X_pos.columns

# Scale
scalers = {}
for col in ['PRCP', 'SNOW', 'TAVG']:
    if col in X_pos.columns:
        scalers[col] = StandardScaler()
        X_pos[col] = scalers[col].fit_transform(X_pos[[col]])

# Pseudo negatives for risk interpretation based on different combinations of factors
X_neg = X_pos.apply(np.random.permutation)
y_neg = np.zeros(len(X_neg))

# Combine for training
X = pd.concat([X_pos, X_neg], axis=0)
y = np.concatenate([y_pos, y_neg])
X, y = shuffle(X, y, random_state=42)

# Add bias for logistic regression
X_lr = np.hstack([np.ones((X.shape[0], 1)), X.values])

# Train/test split
X_train_lr, X_test_lr, y_train, y_test = train_test_split(X_lr, y, test_size=0.2, random_state=42, stratify=y)
X_train_tree, X_test_tree = train_test_split(X, test_size=0.2, random_state=42, stratify=y)

# Train models
lr_model = LogReg(lr=0.1, epochs=1000)
lr_model.fit(X_train_lr, y_train)

tree_model = DecisionTree(max_depth=5)
tree_model.fit(X_train_tree.values, y_train)

# Risk scoring function
def compute_risk_score(input_df: pd.DataFrame) -> np.ndarray:
    # One-hot encode and align columns
    X_input = pd.get_dummies(input_df)
    X_input = X_input.reindex(columns=X_columns, fill_value=0)

    # Apply scalers to continuous features
    for col in ['PRCP', 'SNOW', 'TAVG']:
        if col in X_input.columns:
            X_input[col] = scalers[col].transform(X_input[[col]])

    # Prepare for logistic regression (bias term)
    X_input_values = X_input.values
    X_input_lr = np.hstack([np.ones((len(X_input), 1)), X_input_values])

    # Predict risk scores
    risk_lr = lr_model.predict_prob(X_input_lr)
    risk_tree = tree_model.predict_prob_tree(X_input_values)
    return (risk_lr + risk_tree) / 2

# Risk scoring to original data
data['risk_score'] = compute_risk_score(data[feature_cols])

# Testing purposes
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("\n--- Risk Score Summary ---")
    print("Min:", data['risk_score'].min())
    print("Max:", data['risk_score'].max())
    print("Mean:", data['risk_score'].mean())
    print("Median:", data['risk_score'].median())

    # Histogram
    plt.figure(figsize=(8,5))
    plt.hist(data['risk_score'], bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Risk Scores")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.show()

    # Example new inputs
    sample_conditions = pd.DataFrame([
        {'hour': 8, 'day_of_week': 5, 'is_weekend': 0, 'month': 1, 'time_of_day': 1,
         'road_type': 'divided_highway', 'road_surface': 'dry',
         'PRCP': 0, 'SNOW': 0, 'TAVG': 40, 'weather_category': 'clear',
         'alcohol_related': 0, 'is_freezing': 0},

        {'hour': 18, 'day_of_week': 6, 'is_weekend': 1, 'month': 12, 'time_of_day': 3,
         'road_type': 'residential', 'road_surface': 'wet',
         'PRCP': 0.5, 'SNOW': 0, 'TAVG': 32, 'weather_category': 'rain',
         'alcohol_related': 0, 'is_freezing': 1},
    ])

    sample_risks = compute_risk_score(sample_conditions)
    for i, r in enumerate(sample_risks):
        print(f"Sample {i+1} predicted risk score: {r:.3f}")
