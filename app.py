import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('churn_model.pkl')

st.title("📊 Telco Customer Churn Prediction")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", (0, 1))  # 0=Female, 1=Male
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", (0, 1))
    Partner = st.sidebar.selectbox("Partner", (0, 1))
    Dependents = st.sidebar.selectbox("Dependents", (0, 1))
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.sidebar.selectbox("Phone Service", (0, 1))
    MultipleLines = st.sidebar.selectbox("Multiple Lines", (0, 1, 2))  # No=0, Yes=1, No phone=2
    InternetService = st.sidebar.selectbox("Internet Service", (0, 1, 2))  # DSL=0, Fiber optic=1, No=2
    OnlineSecurity = st.sidebar.selectbox("Online Security", (0, 1, 2))
    OnlineBackup = st.sidebar.selectbox("Online Backup", (0, 1, 2))
    DeviceProtection = st.sidebar.selectbox("Device Protection", (0, 1, 2))
    TechSupport = st.sidebar.selectbox("Tech Support", (0, 1, 2))
    StreamingTV = st.sidebar.selectbox("Streaming TV", (0, 1, 2))
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", (0, 1, 2))
    Contract = st.sidebar.selectbox("Contract", (0, 1, 2))  # Month-to-month=0, One year=1, Two year=2
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", (0, 1))
    PaymentMethod = st.sidebar.selectbox("Payment Method", (0, 1, 2, 3))
    MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2500.0)

    # เตรียมข้อมูลให้ครบทุกคอลัมน์ตามที่โมเดลฝึกไว้
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
churn_labels = ["No Churn", "Churn"]
st.write(churn_labels[prediction[0]])

st.subheader("Prediction Probability")
st.write(prediction_proba)
