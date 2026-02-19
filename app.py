import streamlit as st
import numpy as np
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="centered"
)

# ================================
# LOAD MODEL
# ================================
model = joblib.load("model.pkl")

# ================================
# HEADER
# ================================
st.markdown(
    """
    <h1 style='text-align: center;'>Student Marks Predictor</h1>
    <p style='text-align: center; font-size:18px;'>
    Predict exam score based on study hours using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ================================
# INPUT SECTION
# ================================

col1, col2 = st.columns([1, 1])

with col1:
    hours = st.slider(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        step=0.5
    )

with col2:
    st.markdown("### Info")
    st.write("• Maximum study hours per day: 24")
    st.write("• Maximum exam score: 100")

st.divider()

# ================================
# PREDICTION SECTION
# ================================

if st.button("Predict Score", use_container_width=True):

    if hours < 0 or hours > 24:
        st.error("❌ Study hours must be between 0 and 24.")
    else:
        prediction = model.predict(np.array([[hours]]))[0][0]

        # Cap prediction between 0 and 100
        prediction = max(0, min(prediction, 100))

        st.success("Prediction Generated Successfully!")

        st.metric(
            label=" Predicted Score",
            value=f"{prediction:.2f}"
        )

        st.progress(int(prediction))

st.divider()

# ================================
# FOOTER
# ================================
st.markdown(
    """
    <p style='text-align: center; font-size:14px;'>
    Built with using Streamlit & Scikit-Learn
    </p>
    """,
    unsafe_allow_html=True
)
