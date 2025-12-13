import numpy as np
import joblib
import sklearn   # explicitly using sklearn
from sklearn.linear_model import LogisticRegression

# Load trained sklearn model
model = joblib.load("trained_study_hour_LR_model.pkl")

# Input: study hours
X = np.array([[5]])   # example: 5 hours

# Prediction
prediction = model.predict(X)

print("Predicted Output:", prediction)
