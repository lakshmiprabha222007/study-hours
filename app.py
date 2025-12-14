import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("trained_study_hour_LR_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📘 Study Hours Prediction App")

st.write("Enter the number of study hours to predict marks")

# User input
hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)

if st.button("Predict"):
    hours_array = np.array([[hours]])
    prediction = model.predict(hours_array)
    st.success(f"Predicted Marks: {prediction[0]:.2f}")
