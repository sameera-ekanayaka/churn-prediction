"""
Customer Churn Predictor
-------------------------
Loads the trained Logistic Regression model + fitted StandardScaler
from the churn-prediction project and serves live predictions from
user-entered customer details.

Run with:  streamlit run app.py
(from the churn-prediction project root, so the data/ folder is found)
"""

import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------
# 1. Load model + scaler once
# -----------------------------------------------------------------------
# st.cache_resource ensures these are only loaded once per session,
# not re-loaded from disk on every single widget interaction.

@st.cache_resource
def load_artifacts():
    model = joblib.load("data/log_reg_baseline_model.pkl")
    scaler = joblib.load("data/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# This is the exact column order the model/scaler were trained on.
# Must match X_train.columns.tolist() exactly - order matters.
TRAINING_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'Contract', 'PaperlessBilling', 'MonthlyCharges',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# -----------------------------------------------------------------------
# 2. Page setup
# -----------------------------------------------------------------------
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details to get a live churn prediction from the "
    "trained Logistic Regression model."
)

# -----------------------------------------------------------------------
# 3. Collect raw inputs
# -----------------------------------------------------------------------
st.subheader("Demographics")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
with col2:
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

st.subheader("Account")
col3, col4 = st.columns(2)
with col3:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
with col4:
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5
    )
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

st.subheader("Services")
phone_service = st.selectbox("Phone Service", ["No", "Yes"])

# MultipleLines depends on PhoneService - only meaningful if phone service exists
if phone_service == "Yes":
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
else:
    multiple_lines = "No phone service"
    st.caption("Multiple Lines: No phone service (auto-set, no phone service selected)")

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Internet-dependent add-ons only shown if the customer actually has internet
internet_addon_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
addon_values = {}

if internet_service == "No":
    st.caption("Internet add-ons: all set to 'No internet service' (auto-set, no internet selected)")
    for col in internet_addon_cols:
        addon_values[col] = "No internet service"
else:
    addon_col1, addon_col2 = st.columns(2)
    addon_widgets = [
        ("OnlineSecurity", "Online Security"),
        ("OnlineBackup", "Online Backup"),
        ("DeviceProtection", "Device Protection"),
        ("TechSupport", "Tech Support"),
        ("StreamingTV", "Streaming TV"),
        ("StreamingMovies", "Streaming Movies"),
    ]
    for i, (col, label) in enumerate(addon_widgets):
        target_col = addon_col1 if i % 2 == 0 else addon_col2
        with target_col:
            addon_values[col] = st.selectbox(label, ["No", "Yes"], key=col)

# -----------------------------------------------------------------------
# 4. Encode raw inputs into the model's expected column structure
# -----------------------------------------------------------------------
def encode_input():
    """
    Builds a single-row DataFrame matching TRAINING_COLUMNS exactly -
    same binary map, same ordinal encoding, same one-hot structure
    used in Section 8.2 of the notebook.
    """
    row = {col: 0 for col in TRAINING_COLUMNS}

    # Binary 0/1 columns
    row["gender"] = 1 if gender == "Male" else 0
    row["SeniorCitizen"] = 1 if senior_citizen == "Yes" else 0
    row["Partner"] = 1 if partner == "Yes" else 0
    row["Dependents"] = 1 if dependents == "Yes" else 0
    row["PhoneService"] = 1 if phone_service == "Yes" else 0
    row["PaperlessBilling"] = 1 if paperless_billing == "Yes" else 0

    # Numeric pass-through
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly_charges

    # Ordinal
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    row["Contract"] = contract_map[contract]

    # One-hot: MultipleLines
    row[f"MultipleLines_{multiple_lines}"] = 1

    # One-hot: InternetService
    row[f"InternetService_{internet_service}"] = 1

    # One-hot: internet add-ons
    for col in internet_addon_cols:
        row[f"{col}_{addon_values[col]}"] = 1

    # One-hot: PaymentMethod
    row[f"PaymentMethod_{payment_method}"] = 1

    # Build DataFrame with columns in the exact training order
    return pd.DataFrame([row])[TRAINING_COLUMNS]

# -----------------------------------------------------------------------
# 5. Predict
# -----------------------------------------------------------------------
st.divider()

if st.button("Predict Churn Risk", type="primary"):
    input_df = encode_input()

    # Scale using the FITTED scaler - transform only, never fit_transform here.
    # Fitting on a single new row would compute meaningless mean/std from n=1.
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    churn_probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {churn_probability:.1%} probability")
    else:
        st.success(f"✅ Low Churn Risk — {churn_probability:.1%} probability")

    st.progress(min(churn_probability, 1.0))

    # Business framing based on known EDA risk factors
    risk_factors = []
    if contract == "Month-to-month":
        risk_factors.append("Month-to-month contract")
    if tenure < 12:
        risk_factors.append("Low tenure (new customer)")
    if internet_service == "Fiber optic":
        risk_factors.append("Fiber optic internet")
    if monthly_charges > 70:
        risk_factors.append("High monthly charges")
    if internet_service != "No" and addon_values.get("TechSupport") == "No":
        risk_factors.append("No tech support")

    if risk_factors:
        st.write("**Contributing risk factors identified:**")
        for factor in risk_factors:
            st.write(f"- {factor}")
    else:
        st.write("No major known risk factors present in this profile.")
