import numpy as np
import pickle
from flask import Flask, request, jsonify

# --- Configuration and Constants ---
MODEL_PATH = 'trained_study_hour_LR_model.pkl'

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Model Loading ---
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {MODEL_PATH}")
    # Extract model info for logging/debugging
    COEF = model.coef_[0]
    INTERCEPT = model.intercept_
    print(f"Model Coeff: {COEF:.4f}, Intercept: {INTERCEPT:.4f}")
except FileNotFoundError:
    print(f"ERROR: Model file {MODEL_PATH} not found. Prediction endpoint will be disabled.")
    model = None
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

# --- Web Routes ---

@app.route('/')
def home():
    """Simple health check."""
    return (
        "<h1>Study Hour Predictor API</h1>"
        "<p>Send a **POST** request to the `/predict` endpoint with a JSON payload to get a score prediction.</p>"
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""

    if model is None:
        return jsonify({'error': 'Model could not be loaded on the server.'}), 500

    # 1. Get JSON input
    data = request.get_json(silent=True)
    if not data or 'hours' not in data:
        return jsonify({'error': 'Invalid JSON or missing required key: "hours"'}), 400

    # 2. Process Input
    try:
        study_hours = float(data['hours'])
        
        # scikit-learn models expect input as a 2D array: [[feature_value]]
        input_data = np.array([[study_hours]])
        
    except ValueError:
        return jsonify({'error': 'Invalid value for hours. Must be a number.'}), 400

    # 3. Make Prediction
    try:
        prediction = model.predict(input_data)[0]
        predicted_score = round(float(prediction), 2)

    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500

    # 4. Return Result
    response = {
        'study_hours': study_hours,
        'predicted_score': predicted_score,
        'model_version': 'sklearn 1.6.1'
    }

    return jsonify(response)


# --- Run the App ---
if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
