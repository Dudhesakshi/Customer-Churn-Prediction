# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Title
st.title("🔮 Customer Churn Prediction App")
st.markdown("Fill in the customer details below to predict whether they are likely to churn:")

# Load trained model and feature columns
try:
    model = joblib.load('churn_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
except FileNotFoundError:
    st.error("❌ Model files not found! Please make sure 'churn_model.pkl' and 'feature_columns.pkl' are in the same directory as this app.")
    st.stop()

# Input form
def user_input_features():
    CreditScore = st.slider('Credit Score', 300, 850, 600)
    Age = st.slider('Age', 18, 92, 35)
    Tenure = st.slider('Tenure (Years with Bank)', 0, 10, 3)
    Balance = st.number_input('Balance', 0.0, 300000.0, 50000.0)
    NumOfProducts = st.selectbox('Number of Products', [1, 2, 3, 4])
    HasCrCard = st.selectbox('Has Credit Card', [0, 1])
    IsActiveMember = st.selectbox('Is Active Member', [0, 1])
    EstimatedSalary = st.number_input('Estimated Salary', 1000.0, 200000.0, 50000.0)
    Geography_Germany = st.selectbox('Geography: Germany?', [0, 1])
    Geography_Spain = st.selectbox('Geography: Spain?', [0, 1])
    Gender_Male = st.selectbox('Gender: Male?', [0, 1])

    data = {
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Geography_Germany': Geography_Germany,
        'Geography_Spain': Geography_Spain,
        'Gender_Male': Gender_Male
    }
    return pd.DataFrame([data])

# Capture user input
input_df = user_input_features()

# Prediction
if st.button("Predict"):
    # Ensure all expected columns exist
    missing_cols = list(set(feature_columns) - set(input_df.columns))
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[feature_columns]  # reorder columns

    prediction = model.predict(input_df)[0]
    result = "🔴 Customer is likely to churn." if prediction == 1 else "🟢 Customer is not likely to churn."
    
    st.subheader("Prediction Result:")
    st.success(result)
