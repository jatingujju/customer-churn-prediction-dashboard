import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn Dashboard", page_icon="📊", layout="wide")

# =========================
# LOAD MODEL (PIPELINE)
# =========================
model = joblib.load("models/churn_pipeline.pkl")

# =========================
# HEADER
# =========================
st.title("🚀 Customer Churn Prediction")
st.write("AI-powered system to identify at-risk customers")

# =========================
# INPUT FORM
# =========================
st.subheader("🧾 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72)

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)

# =========================
# PREDICTION
# =========================
st.markdown("---")

if st.button("🔍 Predict Churn", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    # =========================
    # FEATURE ENGINEERING (FIXED)
    # =========================
    input_data['EngagementScore'] = (
        (input_data['OnlineSecurity'] == 'Yes').astype(int) +
        (input_data['OnlineBackup'] == 'Yes').astype(int) +
        (input_data['DeviceProtection'] == 'Yes').astype(int) +
        (input_data['TechSupport'] == 'Yes').astype(int) +
        (input_data['StreamingTV'] == 'Yes').astype(int) +
        (input_data['StreamingMovies'] == 'Yes').astype(int)
    )

    input_data['Recency'] = 1 / (input_data['tenure'] + 1)
    input_data['UsageIntensity'] = input_data['MonthlyCharges'] / (input_data['tenure'] + 1)
    input_data['PriceRatio'] = input_data['MonthlyCharges'] / (input_data['TotalCharges'] + 1)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # =========================
    # DISPLAY RESULT
    # =========================
    st.subheader("📊 Prediction Result")

    st.metric("Churn Probability", f"{prob*100:.2f}%")
    st.progress(prob)

    if prob > 0.6:
        st.error("⚠️ HIGH RISK CUSTOMER")
    elif prob > 0.35:
        st.warning("⚡ MEDIUM RISK CUSTOMER")
    else:
        st.success("✅ LOW RISK CUSTOMER")

    # =========================
    # BUSINESS INSIGHTS
    # =========================
    st.subheader("🧠 AI Insights")

    insights = []

    if tenure < 12:
        insights.append("Low tenure → new customers churn more")

    if MonthlyCharges > 70:
        insights.append("High monthly charges → price sensitivity")

    if Contract == "Month-to-month":
        insights.append("No long-term contract → high churn risk")

    if TechSupport == "No":
        insights.append("No tech support → dissatisfaction risk")

    for i in insights:
        st.info(i)

    # =========================
    # ACTION
    # =========================
    st.subheader("🎯 Recommended Action")

    if prob > 0.6:
        st.error("Offer discount / retention call immediately")
    elif prob > 0.35:
        st.warning("Send engagement offers or reminders")
    else:
        st.success("Maintain good service")