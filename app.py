import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

st.title("Study Hours Prediction App")

# Load trained sklearn model using pickle
with open("trained_study_hour_LR_model.pkl", "rb") as file:
    model = pickle.load(file)

# User input
hours = st.number_input("Enter study hours", min_value=0.0, step=1.0)

if st.button("Predict"):
    prediction = model.predict(np.array([[hours]]))
    st.success(f"Prediction: {prediction[0]}")
