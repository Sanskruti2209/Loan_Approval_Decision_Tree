import streamlit as st
import pandas as pd
import joblib
import os

# Set working directory
os.chdir(r"C:\Users\ACER\OneDrive\Desktop\PGM_PROJECT")
print(f"Current working directory: {os.getcwd()}")  # Debugging output

# Load the pre-trained model
MODEL_PATH = "loan_approval_model.pkl"

# Verify the model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")  # Debugging output

# Streamlit app
st.title("Loan Approval Prediction")
st.write("Enter the applicant's details to predict loan approval.")

# Input fields for features
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0)
loan_amount_term = st.selectbox("Loan Amount Term (days)", [360, 180, 480, 300, 240, 120])
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Prediction button
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 0 if education == "Graduate" else 1,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': {"Urban": 0, "Semiurban": 1, "Rural": 2}[property_area]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Display result
    result = "Approved" if prediction == 1 else "Denied"
    confidence = probability[prediction] * 100
    st.write(f"Prediction: **{result}**")
    st.write(f"Confidence: {confidence:.2f}%")