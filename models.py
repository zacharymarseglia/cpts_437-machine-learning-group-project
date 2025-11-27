"""
Tacoma Crash Prediction - Models Used
- Logistic Regression Model
- Decision Tree

Author: Kendall Reid
"""

import numpy as np

# Logistic Regression model
class LogReg:
    def __init__(self, lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for ep in range(self.epochs):
            z = np.dot(X, self.weights)  # Weighted sum of inputs 
            y_pred = self.sigmoid(z)
            gradient = np.dot(X.T, (y_pred - y)) / n_samples
            self.weights -= self.lr * gradient

    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights)
        return linear_model