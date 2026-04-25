import streamlit as st
import requests

st.title("Customer Churn Prediction")

# Input fields (example)
tenure = st.number_input("Tenure")
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

# Add more fields as per dataset...

if st.button("Predict"):
    data = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['result']}")
    else:
        st.error("Error in prediction")