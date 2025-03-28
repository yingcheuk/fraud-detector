# fraud-detector  

Detecting fraudulent auto insurance claims using structured data and machine learning (Logistic Regression, XGBoost), with a focus on model interpretability and performance evaluation.  

**Goal:** Identify likely fraud cases to help reduce financial losses for insurance providers.  

---

## 📌 Project Overview  

This project builds a machine learning pipeline to identify potentially fraudulent auto insurance claims. It uses only structured tabular data (e.g., customer info, claim history, vehicle details) to predict fraud. The workflow includes data cleaning, outlier segmentation, model comparison, and evaluation.  

---

## 🧠 Techniques Used

- Supervised Learning: Logistic Regression, XGBoost  
- Data Cleaning: Outlier removal (IQR), inconsistent date handling  
- Feature Encoding: One-hot and Label Encoding  
- Scaling: StandardScaler for numerical features  
- Evaluation: Precision, Recall, F1-score, ROC AUC  
- (planned) Explainability: SHAP values for model interpretation  

---

## 📁 Project Structure

`/data` — Raw and cleaned CSVs   
`/notebooks` — EDA, modeling, evaluation  
`/src` — Modular code (e.g., preprocessing, utils)  
`/models` — Saved model files (e.g., `.pkl`)  
`README.md` — You are here 📄  

---

## 🚀 Current Goals

- Build and compare models for both inlier and outlier segments  
- Select the best-performing model per segment  
- Ensure reproducibility and modular pipeline design  
- Prepare for adding explainability tools like SHAP  

---

## 🔮 Future Improvements

- Incorporate textual features (if available) for hybrid modeling  
- Add SHAP-based insights for fraud investigator support  
- Build a user-facing app using Streamlit or Gradio  
- Deploy as an API for integration into production  
