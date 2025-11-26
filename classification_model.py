"""
Tacoma Crash Prediction - Model
-

Author: Kendall Reid
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
import models as md


# Load data
data = pd.read_csv("cleaned_tacoma_crashes_with_weather.csv")


# Feature selection
feature_cols = [
    'hour', 'day_of_week', 'is_weekend', 'month',
    'time_of_day', 'road_type', 'road_surface',
    'PRCP', 'SNOW', 'TAVG', 'weather_category',
    'alcohol_related', 'is_freezing'
]
target_col = 'crash_occurred'

x_pos = pd.get_dummies(data[feature_cols], drop_first=True)  # One hot encoding of relevant categorical features
y_pos = np.ones(len(x_pos))  # All crahses so label all 1


# Create pseudo-negatives
n_neg = len(x_pos)
x_neg = x_pos.apply(np.random.permutation)
y_neg = np.zeros(n_neg)


# Combine positives and negatives into one large dataset
X = pd.concat([x_pos, x_neg], axis = 0)
y = np.concatenate([y_pos, y_neg])
X, y = shuffle(X, y, random_state=42)

X_lr = np.hstack([np.ones((X.shape[0], 1)), X.values])  # Add bias term for logistic regression


# Train/test split
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_sixe = 0.2, random_state = 43, stratify = y)
X_train_tree, X_test_tree = train_test_split(X, test_size = 0.3, random_state = 43, stratify = y)

