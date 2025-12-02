"""
Model Loader Utility
Loads trained models for risk score calculation in Flask app.
This allows the models to be used without retraining each time.

Author: Zuriel H (based on Kendall Reid's crash_prediction.py)
"""

import pandas as pd
import numpy as np
import pickle
import os
from models import LogReg, DecisionTree
from sklearn.preprocessing import StandardScaler

# Model file paths
MODEL_DIR = 'models_saved'
LR_MODEL_FILE = os.path.join(MODEL_DIR, 'lr_model.pkl')
TREE_MODEL_FILE = os.path.join(MODEL_DIR, 'tree_model.pkl')
SCALERS_FILE = os.path.join(MODEL_DIR, 'scalers.pkl')
X_COLUMNS_FILE = os.path.join(MODEL_DIR, 'x_columns.pkl')

# Global variables for loaded models
_lr_model = None
_tree_model = None
_scalers = None
_x_columns = None


def save_models(lr_model, tree_model, scalers, x_columns):
    """
    Save trained models and preprocessing objects to disk.
    
    Args:
        lr_model: Trained LogReg model
        tree_model: Trained DecisionTree model
        scalers: Dictionary of StandardScaler objects
        x_columns: Column names from training data
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save models (custom classes need special handling)
    with open(LR_MODEL_FILE, 'wb') as f:
        pickle.dump(lr_model.weights, f)
        pickle.dump(lr_model.lr, f)
        pickle.dump(lr_model.epochs, f)
    
    with open(TREE_MODEL_FILE, 'wb') as f:
        pickle.dump(tree_model.tree, f)
        pickle.dump(tree_model.max_depth, f)
        pickle.dump(tree_model.min_samples_split, f)
    
    with open(SCALERS_FILE, 'wb') as f:
        pickle.dump(scalers, f)
    
    with open(X_COLUMNS_FILE, 'wb') as f:
        pickle.dump(x_columns, f)
    
    print(f"✓ Models saved to {MODEL_DIR}/")


def load_models():
    """
    Load trained models and preprocessing objects from disk.
    
    Returns:
        tuple: (lr_model, tree_model, scalers, x_columns) or None if not found
    """
    global _lr_model, _tree_model, _scalers, _x_columns
    
    # Return cached models if already loaded
    if _lr_model is not None:
        return _lr_model, _tree_model, _scalers, _x_columns
    
    if not os.path.exists(LR_MODEL_FILE):
        return None
    
    try:
        # Load logistic regression model
        with open(LR_MODEL_FILE, 'rb') as f:
            weights = pickle.load(f)
            lr = pickle.load(f)
            epochs = pickle.load(f)
        
        _lr_model = LogReg(lr=lr, epochs=epochs)
        _lr_model.weights = weights
        
        # Load decision tree model
        with open(TREE_MODEL_FILE, 'rb') as f:
            tree = pickle.load(f)
            max_depth = pickle.load(f)
            min_samples_split = pickle.load(f)
        
        _tree_model = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        _tree_model.tree = tree
        
        # Load scalers
        with open(SCALERS_FILE, 'rb') as f:
            _scalers = pickle.load(f)
        
        # Load column names
        with open(X_COLUMNS_FILE, 'rb') as f:
            _x_columns = pickle.load(f)
        
        print("✓ Models loaded successfully")
        return _lr_model, _tree_model, _scalers, _x_columns
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None


def compute_risk_score(input_df: pd.DataFrame) -> float:
    """
    Compute risk score for given input conditions.
    Uses loaded models if available, otherwise returns None.
    
    Args:
        input_df: DataFrame with feature columns matching training data
        
    Returns:
        float: Risk score (0-1) or None if models not loaded
    """
    models = load_models()
    if models is None:
        return None
    
    lr_model, tree_model, scalers, x_columns = models
    
    try:
        # One-hot encode and align columns
        X_input = pd.get_dummies(input_df)
        X_input = X_input.reindex(columns=x_columns, fill_value=0)
        
        # Apply scalers to continuous features
        for col in ['PRCP', 'SNOW', 'TAVG']:
            if col in X_input.columns and col in scalers:
                X_input[col] = scalers[col].transform(X_input[[col]])
        
        # Prepare for logistic regression (bias term)
        X_input_values = X_input.values
        X_input_lr = np.hstack([np.ones((len(X_input), 1)), X_input_values])
        
        # Predict risk scores
        risk_lr = lr_model.predict_prob(X_input_lr)
        risk_tree = tree_model.predict_prob_tree(X_input_values)
        risk_score = float((risk_lr[0] + risk_tree[0]) / 2)
        
        return risk_score
        
    except Exception as e:
        print(f"Error computing risk score: {e}")
        return None

