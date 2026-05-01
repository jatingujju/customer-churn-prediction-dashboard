import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details:")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure", 1, 72)
MonthlyCharges = st.number_input("Monthly Charges", 20.0, 120.0)
TotalCharges = st.number_input("Total Charges", 100.0, 8000.0)
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
SupportTickets = st.slider("Support Tickets", 0, 10)

# Create input dataframe
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "tenure": [tenure],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
    "Contract": [Contract],
    "InternetService": [InternetService],
    "SupportTickets": [SupportTickets]
})

# Encode input
for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Churn ({prob:.2f})")