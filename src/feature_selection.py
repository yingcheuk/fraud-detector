"""
Feature Selection Utilities for XGBoost Models

This module provides helper functions to extract and filter the most important features 
based on feature importance scores (e.g., gain) from trained XGBoost models. 

The goal is to identify high-impact features that may contribute most to model performance 
and optionally reduce noise by excluding less useful features.
"""


def get_important_features_from_xgb(model, feature_names, threshold=0.5, return_names=True):
    """
    Extract important features from an XGBoost model using gain importance.

    Parameters:
    - model: Trained XGBClassifier model
    - feature_names: List of original feature names
    - threshold: Minimum gain value to be considered important
    - return_names: If True, return list of feature names; else return indices

    Returns:
    - List of important feature names or indices
    """
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type='gain')

    important_indices = [int(f[1:]) for f, g in score_dict.items() if g >= threshold]

    if return_names:
        return [feature_names[i] for i in important_indices]
    else:
        return important_indices


