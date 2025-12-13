import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

# Load model
loaded_model = joblib.load("model.pkl")

# Test prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = loaded_model.predict(sample)

print("Prediction:", prediction)
