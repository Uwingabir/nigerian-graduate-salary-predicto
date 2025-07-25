import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Create a simple mock model for testing
print("Creating simple test model...")

# Create a basic linear regression model
model = LinearRegression()

# Create some dummy training data
X_dummy = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y_dummy = np.array([100000, 200000, 300000, 400000])

# Train the model
model.fit(X_dummy, y_dummy)

# Save the model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Simple test model created!")

# Test loading
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

test_pred = loaded_model.predict([[5, 6, 7]])[0]
print(f"Test prediction: {test_pred}")

print("✅ Model save/load test successful!")
