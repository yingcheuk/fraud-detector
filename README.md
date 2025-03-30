# 🕵️‍♀️ fraud-detector

Detecting fraudulent auto insurance claims using machine learning (Logistic Regression, XGBoost) with a strong focus on **model performance, interpretability, and pipeline modularity**.


**Goal:** Accurately flag potentially fraudulent claims to help insurers reduce financial losses.

---

## 📌 Project Overview

This project builds a machine learning pipeline to detect fraud in structured auto insurance claim data. It includes:  
- Cleaning & preprocessing raw input data  
- Handling inliers vs. outliers separately  
- Training and comparing multiple models  
- Generating predictions for new data  
- Laying the groundwork for future explainability  

---

## 💡 Why This Matters  
Insurance fraud is a high-impact use case for ML. This project shows how:  
- Domain-specific preprocessing (inliers/outliers) boosts performance  
- Explainability tools like SHAP make black-box models more trustworthy  
- Modular pipelines keep experiments clean and reproducible  

---

## 🧠 Techniques Used

- Supervised Learning: Logistic Regression, XGBoost  
- Outlier Handling: IQR-based removal  
- Feature Engineering: One-hot encoding (LogReg), Ordinal encoding (XGBoost)  
- Feature Scaling: StandardScaler (for Logistic Regression)
- Evaluation Metrics: Accuracy, Precision, Recall, F1, ROC AUC
- Class Imbalance Handling: scale_pos_weight in XGBoost   
- Model Explainability: SHAP (SHapley Additive Explanations)  

---

## 📁 Project Structure
```
fraud-detector/
│
├── data/                   # Input data and prediction results
├── models/                 # Saved models and preprocessors (.pkl)
├── notebooks/              # EDA, modeling, and inference notebooks
├── src/                    # Modular Python code
│   ├── preprocessing.py        # Data cleaning, outlier removal
│   ├── prep_features.py        # Feature encoding and scaling
│   ├── train.py                # Model training and evaluation
│   ├── inference.py            # Batch inference on new data
│   ├── visualization.py        # SHAP explanations and feature name utilities
│   └── feature_selection.py    # Feature filtering (optional experiment)
└── README.md               # Project documentation
```
---

## 🔧 Function Overview

The pipeline is built from reusable, modular functions:  

- 🧼 `clean_data()` & `remove_outliers_iqr()`  
  → Clean the dataset, fix dates, and isolate outliers  

- 📉 `check_class_ratio()`  
  → Explore fraud distribution across inliers and outliers   

- ⚙️ `prepare_for_logreg()` & `prepare_for_xgb()`  
  → Preprocess features for each model type (scaling + encoding)

- 🏷️ `get_feature_names_from_column_transformer()`
  → Extract readable feature names from fitted ColumnTransformer  

- 🧠 `train_models()`  
  → Train four models (inliers/outliers × LR/XGB) and save them  

- 📈 `build_summary_table()`  
  → Tabulate model performance across key metrics  

- 🔍 `predict_new_data()`  
  → Run saved models on new incoming claims for fraud prediction

- 🧠 `explain_model_with_shap()`  
  → Visualize top drivers of XGBoost model predictions using SHAP    

---

### 📒 Notebooks

- `01_data_exploration.ipynb`: Clean raw data, explore distributions, and split into inliers/outliers  
- `02_modeling.ipynb`: Train and compare 4 models (LR & XGB on inliers/outliers)  
- `03_inference.ipynb`: Apply trained models to new test claims and generate predictions    

---

## 🚀 Current Goals

- Build interpretable and modular fraud detection models  
- Compare performance across inliers and outliers  
- Optimize for recall and precision to balance false positives  
- Prepare for downstream tasks like SHAP-based interpretation  

---

### 📤 Example Output

After running inference on new claims, you get an output CSV like:  

| policy_number | incident_date | ... | total_claim_amount | fraud_predicted |
|---------------|----------------|-----|---------------------|------------------|
| 54321         | 2023-01-14     | ... | 8200.0              | 1                |
| 67890         | 2023-02-11     | ... | 3100.0              | 0                |

---

## 🔮 Future Improvements

- Add model-based alert thresholds for production deployment  
- Test hybrid models using both structured and textual features  
- Deploy via Streamlit or Gradio for user-friendly interaction  
- Package the pipeline into an API for production integration
- Automate model monitoring and threshold adjustment  
