"""
Tacoma Crash Prediction - Model
-

Author: Kendall Reid
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score


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

X = pd.get_dummies(data[feature_cols], drop_first=True) # One hot encoding of relevant categorical features
Y = data[target_col].values
print(Y)
print("hello")