# fraud-detector

Detecting fraudulent insurance claims using structured data and NLP techniques (XGBoost + SHAP).

**Goal:** Detect fraudulent auto insurance claims using structured and textual data.  
Build a machine learning model that can identify likely fraud cases and help reduce financial loss for insurance providers.  

---

## 📌 Project Overview
This project aims to build a machine learning model that identifies potentially fraudulent insurance claims. It combines structured data (e.g., claim amounts, customer info) and unstructured data (text descriptions) to predict the likelihood of fraud.

---

## 🧠 Techniques Used
- Supervised Learning: Logistic Regression, XGBoost
- NLP: Text preprocessing, TF-IDF
- Explainability: SHAP values
- Evaluation: Precision, Recall, F1-score, ROC AUC

---

## 📁 Project Structure

`/data` — Raw and processed data  
`/notebooks` — Exploratory analysis and model development  
`/src` — Source code (preprocessing, modeling)  
`/models` — Saved trained models  
`README.md` — Project documentation  

---

## 🚀 Goals

- Build an interpretable fraud detection model
- Combine text + structured features
- Handle class imbalance
- Deliver actionable insights for fraud analysts

---

## 🔮 Future Improvements
- Use BERT embeddings for better NLP
- Deploy as a Streamlit web app
- Integrate with real-time scoring pipeline
