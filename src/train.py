"""
Model Training:

This script trains two types of models (Logistic Regression and XGBoost)
on both inlier and outlier datasets.

Steps:
- Fit models using inliers and outliers separately
- Evaluate all models using classification_report (focus on recall and F1-score)
- Select one best model for inliers and one for outliers
- Save the final selected models to the models/ directory
"""

# function 1: model training

from sklearn.metrics import classification_report
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd
from prep_features import prepare_for_logreg, prepare_for_xgb

def train_models(df_inliers, df_outliers, detail = False, save_path='/Users/jennifercyc/Desktop/HSH/ML Project/fraud-detector/models/', random_state=42):
    reports = []
    metrics_data = []

    # --- Model 1: Logistic Regression (inliers) ---
    X1, y1, preprocessor1 = prepare_for_logreg(df_inliers, return_preprocessor=True)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, stratify=y1, random_state=random_state)
    model1 = LogisticRegression(max_iter=1000)
    # Train model
    model1.fit(X1_train, y1_train)
    y1_pred = model1.predict(X1_test)
    y1_proba = model1.predict_proba(X1_test)[:, 1]
    reports.append(classification_report(y1_test, y1_pred, output_dict=True))
    metrics_data.append((y1_test, y1_pred, y1_proba))
    if detail:
        print("\nModel 1: LR Inliers\n", classification_report(y1_test, y1_pred))
    # Save model
    joblib.dump(model1, f"{save_path}model1_lr_inliers.pkl")
    joblib.dump(preprocessor1, f"{save_path}preprocessor1_inliers.pkl")

    
    # --- Model 2: XGBoost (inliers) ---
    X2, y2, preprocessor2, feature_names2 = prepare_for_xgb(df_inliers, return_preprocessor=True)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, stratify=y2, random_state=random_state)
    model2 = XGBClassifier(eval_metric='logloss')
    # Train model
    model2.fit(X2_train, y2_train)
    y2_pred = model2.predict(X2_test)
    y2_proba = model2.predict_proba(X2_test)[:, 1]
    reports.append(classification_report(y2_test, y2_pred, output_dict=True))
    metrics_data.append((y2_test, y2_pred, y2_proba))
    if detail:
        print("\nModel 2: XGB Inliers\n", classification_report(y2_test, y2_pred))
    # Save model
    joblib.dump(model2, f"{save_path}model2_xgb_inliers.pkl")
    joblib.dump(preprocessor2, f"{save_path}preprocessor2_inliers.pkl")
    joblib.dump(X2_test, f"{save_path}X2_test_inliers.pkl")

    
    # --- Model 3: Logistic Regression (outliers) ---
    X3, y3, preprocessor3 = prepare_for_logreg(df_outliers, return_preprocessor=True)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, stratify=y3, random_state=random_state)
    model3 = LogisticRegression(max_iter=1000)
    # Train model
    model3.fit(X3_train, y3_train)
    y3_pred = model3.predict(X3_test)
    y3_proba = model3.predict_proba(X3_test)[:, 1]
    reports.append(classification_report(y3_test, y3_pred, output_dict=True))
    metrics_data.append((y3_test, y3_pred, y3_proba))
    if detail:
        print("\nModel 3: LR Outliers\n", classification_report(y3_test, y3_pred))
    # Save model
    joblib.dump(model3, f"{save_path}model3_lr_outliers.pkl")
    joblib.dump(preprocessor3, f"{save_path}preprocessor3_outliers.pkl")

    
    # --- Model 4: XGBoost (outliers) ---
    X4, y4, preprocessor4, feature_names4 = prepare_for_xgb(df_outliers, return_preprocessor=True)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, stratify=y4, random_state=random_state)
    model4 = XGBClassifier(eval_metric='logloss')
    # Train model
    model4.fit(X4_train, y4_train)
    y4_pred = model4.predict(X4_test)
    y4_proba = model4.predict_proba(X4_test)[:, 1]
    reports.append(classification_report(y4_test, y4_pred, output_dict=True))
    metrics_data.append((y4_test, y4_pred, y4_proba))
    if detail:
        print("\nModel 4: XGB Outliers\n", classification_report(y4_test, y4_pred))
    # Save model
    joblib.dump(model4, f"{save_path}model4_xgb_outliers.pkl")
    joblib.dump(preprocessor4, f"{save_path}preprocessor4_outliers.pkl")
    joblib.dump(X4_test, f"{save_path}X4_test_outliers.pkl")

    return reports, metrics_data



# function 2: compare the models

from sklearn.metrics import roc_auc_score

def build_summary_table(reports, metrics_data, save_path='/Users/jennifercyc/Desktop/HSH/ML Project/fraud-detector/models/'):
    
    model_names = [
        "Model 1: LR Inliers",
        "Model 2: XGB Inliers",
        "Model 3: LR Outliers",
        "Model 4: XGB Outliers"
    ]

    metrics = {
        "F1 (Fraud)":        [report['1']['f1-score']    for report in reports],
        "Recall (Fraud)":    [report['1']['recall']      for report in reports],
        "Precision (Fraud)": [report['1']['precision'] for report in reports],
        "Accuracy":          [report['accuracy']         for report in reports],
        "ROC AUC":           [roc_auc_score(y_true, y_proba) for (y_true, _, y_proba) in metrics_data]
    }

    summary_df = pd.DataFrame(metrics, index=model_names).round(2)
    summary_df.to_csv(f"{save_path}model_metrics_summary.csv")
    print(summary_df)
    
    return summary_df
