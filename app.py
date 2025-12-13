import streamlit as st
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

st.title("Study Hours Prediction App")

# Load trained sklearn model
model = joblib.load("trained_study_hour_LR_model.pkl")

# User input
hours = st.number_input("Enter study hours:", min_value=0.0, step=0.5)

if st.button("Predict"):
    X = np.array([[hours]])
    prediction = model.predict(X)
    st.success(f"Prediction: {prediction[0]}")
