"""
Tacoma Crash Prediction - Models
- Logistic Regression Model
- Decision Tree

Author: Kendall Reid
"""

import numpy as np

# Logistic Regression
class LogReg:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    def sigmoid(self, z):
        z = np.array(z, dtype=float)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.epochs):
            z = np.dot(X, self.weights)
            y_pred = self.sigmoid(z)
            gradient = np.dot(X.T, (y_pred - y)) / n_samples
            self.weights -= self.lr * gradient

    def predict_prob(self, X):
        X = np.array(X, dtype=float)
        linear_model = np.dot(X, self.weights)
        return self.sigmoid(linear_model)


# Decision Tree
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value  # probability at leaf

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            prob = np.mean(y)
            return self.Node(value=prob)

        best_feature, best_thresh, best_gain = None, None, -1
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feature], thresh)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = thresh

        if best_gain == 0:
            return self.Node(value=np.mean(y))

        left_idx = X[:, best_feature] <= best_thresh
        right_idx = X[:, best_feature] > best_thresh
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return self.Node(feature=best_feature, threshold=best_thresh, left=left, right=right)

    def _gini(self, y):
        p = np.mean(y)
        return 1 - (p ** 2 + (1 - p) ** 2)

    def _information_gain(self, y, feature_col, threshold):
        parent_gini = self._gini(y)
        left_idx = feature_col <= threshold
        right_idx = feature_col > threshold
        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            return 0
        n = len(y)
        n_left, n_right = len(y[left_idx]), len(y[right_idx])
        child_gini = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])
        return parent_gini - child_gini

    def predict_prob_tree(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _predict_row(self, row, node):
        if node.value is not None:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        else:
            return self._predict_row(row, node.right)
