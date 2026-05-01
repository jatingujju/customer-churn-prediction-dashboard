import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn Dashboard", page_icon="📊", layout="wide")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1 {
    text-align: center;
    color: #4CAF50;
}
.card {
    background: linear-gradient(145deg, #1c1f26, #111318);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid #2c2f36;
}
.metric-box {
    background-color: #262730;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #2ecc71);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# =========================
# HEADER
# =========================
st.markdown("<h1>🚀 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered system to identify at-risk customers</p>", unsafe_allow_html=True)

# =========================
# DASHBOARD SUMMARY
# =========================
st.markdown("### 📊 Quick Overview")
colA, colB, colC = st.columns(3)

with colA:
    st.metric("Model", "Random Forest")

with colB:
    st.metric("Accuracy", "80%")

with colC:
    st.metric("Use Case", "Retention")

# =========================
# INPUT LAYOUT
# =========================
col1, col2 = st.columns(2)

# LEFT
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Customer Info")

    gender = st.selectbox("Gender", ["Male", "Female"])
    st.markdown("<br>", unsafe_allow_html=True)

    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    st.markdown("<br>", unsafe_allow_html=True)

    Partner = st.selectbox("Partner", ["Yes", "No"])
    st.markdown("<br>", unsafe_allow_html=True)

    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    st.markdown("<br>", unsafe_allow_html=True)

    tenure = st.slider("Tenure (Months)", 0, 72)

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💳 Service & Billing")

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

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
st.markdown("---")

if st.button("🚀 Predict Churn", use_container_width=True):

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

    # Encode input
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # RESULT DISPLAY
    st.markdown("<h2 style='text-align:center;'>📊 Prediction Result</h2>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style='text-align:center; padding:20px'>
            <h1 style='font-size:60px; color:#4CAF50'>{prob*100:.1f}%</h1>
            <p>Churn Probability</p>
        </div>
    """, unsafe_allow_html=True)

    st.progress(prob)

    if prob > 0.6:
        st.error("⚠️ HIGH RISK CUSTOMER")
    elif prob > 0.3:
        st.warning("⚡ MEDIUM RISK CUSTOMER")
    else:
        st.success("✅ LOW RISK CUSTOMER")

    # INSIGHTS
    st.markdown("### 🧠 AI Insights")

    insights = []

    if tenure < 12:
        insights.append("Low tenure → New customers tend to churn")

    if MonthlyCharges > 70:
        insights.append("High charges → price sensitivity")

    if Contract == "Month-to-month":
        insights.append("No contract → unstable customer")

    if TechSupport == "No":
        insights.append("No tech support → dissatisfaction risk")

    if insights:
        for i in insights:
            st.info(i)
    else:
        st.success("Customer profile looks stable")

    # ACTIONS
    st.markdown("### 🎯 Recommended Action")

    if prob > 0.6:
        st.error("Offer discount or retention call immediately")
    elif prob > 0.3:
        st.warning("Send engagement offers or reminders")
    else:
        st.success("Maintain customer experience")