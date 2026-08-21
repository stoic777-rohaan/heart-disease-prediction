import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load trained artifacts
# ==========================
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
expected_columns = joblib.load("models/columns.pkl")

# ==========================
# Page setup
# ==========================
st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details below to estimate heart disease risk.")

# ==========================
# User inputs
# ==========================
Age = st.slider("Age", 18, 100, 50)
Sex = st.selectbox("Sex", ["M", "F"])
ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)
Cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 700, 200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
MaxHR = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
Oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.5, 1.0)
ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ==========================
# Prediction
# ==========================
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Age": [Age],
        "Sex": [Sex],
        "ChestPainType": [ChestPainType],
        "RestingBP": [RestingBP],
        "Cholesterol": [Cholesterol],
        "FastingBS": [FastingBS],
        "RestingECG": [RestingECG],
        "MaxHR": [MaxHR],
        "ExerciseAngina": [ExerciseAngina],
        "Oldpeak": [Oldpeak],
        "ST_Slope": [ST_Slope],
    })

    # Same one-hot encoding used during training in the notebook
    input_encoded = pd.get_dummies(input_data)

    # Add any columns the model expects but this input didn't produce
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Keep exact column order the scaler/model were trained on
    input_encoded = input_encoded[expected_columns]

    scaled_input = scaler.transform(input_encoded)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {probability:.2%}")
