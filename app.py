# app.py (ปรับปรุงใหม่)

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('churn_model.pkl')

st.title("📊 Telco Customer Churn Prediction")

def user_input_features():
    tenure = st.sidebar.number_input('Tenure (months)', 0, 72, 12)
    MonthlyCharges = st.sidebar.number_input('Monthly Charges', 0.0, 120.0, 70.0)
    TotalCharges = st.sidebar.number_input('Total Charges', 0.0, 9000.0, 1500.0)
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', (0,1))
    Contract = st.sidebar.selectbox('Contract Type', (0,1,2))
    InternetService = st.sidebar.selectbox('Internet Service', (0,1,2))
    PaymentMethod = st.sidebar.selectbox('Payment Method', (0,1,2,3))

    # สร้าง DataFrame โดยใช้คอลัมน์ทั้งหมดจากตอน train
    columns_from_train = model.feature_names_in_
    input_data = pd.DataFrame(columns=columns_from_train)

    # กำหนดค่า default เป็น 0 ทั้งหมด
    input_data.loc[0] = 0

    # ใส่ค่าเฉพาะที่ผู้ใช้ป้อนเข้ามา
    input_data.at[0, 'tenure'] = tenure
    input_data.at[0, 'MonthlyCharges'] = MonthlyCharges
    input_data.at[0, 'TotalCharges'] = TotalCharges
    input_data.at[0, 'SeniorCitizen'] = SeniorCitizen
    input_data.at[0, 'Contract'] = Contract
    input_data.at[0, 'InternetService'] = InternetService
    input_data.at[0, 'PaymentMethod'] = PaymentMethod

    return input_data

input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
churn_labels = ["No Churn", "Churn"]
st.write(churn_labels[prediction[0]])

st.subheader('Prediction Probability')
st.write(prediction_proba)
