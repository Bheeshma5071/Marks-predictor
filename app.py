import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("🎓 Student Marks Predictor")

# Input from user
hours = st.number_input(
    "Enter study hours (0 - 24):",
    min_value=0.0,
    max_value=24.0,
    step=0.5
)

if st.button("Predict Score"):

    # Extra validation (safety check)
    if hours < 0 or hours > 24:
        st.error("❌ Study hours must be between 0 and 24.")
    else:
        prediction = model.predict(np.array([[hours]]))[0][0]

        # Ensure score is within 0-100
        prediction = max(0, min(prediction, 100))

        st.success(f"📊 Predicted Score: {prediction:.2f}")
