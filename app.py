import streamlit as st
import numpy as np
import pickle

# --- Configuration and Constants ---
# MUST be the exact filename of your model file
MODEL_PATH = 'trained_study_hour_LR_model.pkl'

# --- Model Loading (using caching for performance) ---
# @st.cache_resource is used to load the model only once
@st.cache_resource
def load_model():
    """
    Loads the pickled scikit-learn model and handles errors.
    """
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please verify the filename and path in your GitHub repository.")
        return None
    except Exception as e:
        # This is where the 'No module named sklearn' error occurs
        st.error(f"Error loading model. This usually means a missing dependency (scikit-learn). Full error: {e}")
        st.warning("Please ensure your 'requirements.txt' file contains 'scikit-learn==1.6.1' and has been committed to GitHub.")
        return None

# Load the model globally
model = load_model()

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Study Hour Score Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Predictive Modeling Demo: Study Hours to Score")
st.markdown("---")

if model is not None:
    # --- Input Section ---
    st.header("1. Input Study Hours")
    
    study_hours = st.slider(
        'Select the number of hours studied:',
        min_value=0.0,
        max_value=20.0,
        value=5.0, # Default value
        step=0.1,
        help="Adjust the slider to see how study hours impact the predicted score."
    )
    
    st.info(f"Selected study time: **{study_hours:.1f} hours**")
    
    # --- Prediction Logic ---
    st.header("2. Predicted Exam Score")
    
    if st.button('Calculate Predicted Score', use_container_width=True):
        try:
            # Prepare the input data for the scikit-learn model
            input_data = np.array([[study_hours]])
            
            # Make the prediction
            prediction = model.predict(input_data)[0]
            
            # Format the output
            predicted_score = round(float(prediction), 2)
            
            # Display the result
            st.success(f"Based on **{study_hours:.1f} hours** of study, the predicted exam score is:")
            st.balloons()
            st.metric(label="Predicted Score", value=f"{predicted_score}")

            with st.expander("Show Model Details"):
                st.write(f"**Model Type:** Linear Regression (sklearn 1.6.1)")
                st.write(f"**Coefficient (Impact per Hour):** {model.coef_[0]:.2f}")
                st.write(f"**Intercept (Base Score):** {model.intercept_:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("The predictive model could not be loaded. Please fix the errors listed above.")

st.markdown("---")
st.caption("App built with Streamlit.")
