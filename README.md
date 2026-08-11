# Customer Churn Prediction

A machine learning project predicting customer churn for a telecom provider, built as hands-on preparation for a CRM system (Senior Design Project) that includes churn prediction and RFM segmentation as core features.

## Overview

This project covers the full pipeline from raw data to a live, interactive prediction tool:

- Exploratory Data Analysis to uncover churn drivers
- Feature engineering (encoding, multicollinearity handling)
- Model training and comparison (Logistic Regression vs. Random Forest)
- A Streamlit app serving live predictions from the trained model

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Key Findings (EDA)

Churn clusters strongly around customers who are:
- New (low tenure)
- On month-to-month contracts
- Using fiber optic internet
- Paying high monthly charges
- Without tech support

## Modeling

Two models were trained and compared on the same engineered feature set:

| Model | Recall (Churn) | Precision (Churn) | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.51 | 0.61 | 0.74 |
| Random Forest | 0.60 | 0.53 | 0.56 | 0.75 |

**Selected model: Logistic Regression**, chosen for its higher recall — in a churn-prevention context, catching more actual churners matters more than avoiding false positives, since the cost of missing an at-risk customer (lost revenue) typically outweighs the cost of an unnecessary retention offer. Logistic Regression's interpretability was a secondary factor in this choice.

Class imbalance (~73.5% retained / ~26.5% churned) was handled via `class_weight='balanced'` rather than SMOTE, to avoid introducing synthetic data into a first-pass model.

## Live Predictor

An interactive Streamlit app (`app.py`) loads the trained model and lets you input a customer's details to get a live churn prediction and risk breakdown.

### Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
churn-prediction/
├── data/                    # Dataset, checkpoints, trained model/scaler
├── notebooks/
│   └── churn_prediction.ipynb   # Full EDA -> feature engineering -> modeling pipeline
├── app.py                   # Streamlit live predictor
├── requirements.txt
└── README.md
```

## Tech Stack

Python, pandas, scikit-learn, Streamlit, matplotlib/seaborn, joblib

## Related Project

Paired with a separate RFM customer segmentation project (repo linked once complete), both feeding into the broader CRM system SDP.
