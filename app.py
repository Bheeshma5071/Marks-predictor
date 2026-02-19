import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Page title
st.title("📊 Student Marks Predictor")

st.write("Enter number of study hours to predict marks.")

# Input from user
hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)

# Prediction button
if st.button("Predict Score"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Score: {prediction[0]:.2f}")
