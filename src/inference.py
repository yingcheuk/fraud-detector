"""
Model Inference:
Apply the trained XGBoost models to new, unseen data to predict whether each insurance claim is likely to be fraudulent or not.
"""

import os
import pandas as pd
import joblib
from preprocessing import clean_data, remove_outliers_iqr
from prep_features import prepare_for_logreg, prepare_for_xgb


def predict_new_data(
    filename,
    inlier_model="model2_xgb_inliers.pkl",
    outlier_model="model4_xgb_outliers.pkl",
    data_dir="/Users/jennifercyc/Desktop/HSH/ML Project/fraud-detector/data/",
    model_dir="/Users/jennifercyc/Desktop/HSH/ML Project/fraud-detector/models/",
    output_path="/Users/jennifercyc/Desktop/HSH/ML Project/fraud-detector/data/predictions.csv"
):
    # Construct full paths
    file_path = os.path.join(data_dir, filename)
    inlier_model_path = os.path.join(model_dir, inlier_model)
    outlier_model_path = os.path.join(model_dir, outlier_model)

    # Load data and models
    ext = os.path.splitext(file_path)[-1]       #determine file type
    if ext == ".xlsx":
        df_raw = pd.read_excel(file_path)
    elif ext == ".csv":
        df_raw = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format: must be .csv or .xlsx")
    
    df_clean = clean_data(df_raw)
    df_inliers, df_outliers = remove_outliers_iqr(df_clean)
    
    model_inlier = joblib.load(inlier_model_path)
    model_outlier = joblib.load(outlier_model_path)

    # Prepare features
    X_in = prepare_for_xgb(df_inliers, return_target=False)
    X_out = prepare_for_xgb(df_outliers, return_target=False)

    # Predict
    preds_in = model_inlier.predict(X_in)
    preds_out = model_outlier.predict(X_out)

    # Combine
    df_inliers = df_inliers.copy()
    df_outliers = df_outliers.copy()
    df_inliers.loc[:, 'fraud_predicted'] = preds_in
    df_outliers.loc[:, 'fraud_predicted'] = preds_out

    df_combined = pd.concat([df_inliers, df_outliers]).sort_index()

    # Save
    df_combined.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")

    return df_combined