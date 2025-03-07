from pysr import PySRRegressor  # Symbolic Regression library
import numpy as np

# Load collected data
data = np.load("collected_policy_data.npz")
X = data["observations"]  # States
y = data["actions"]  # Actions

# Train symbolic regression model
model = PySRRegressor(
    n_iterations=40,
    populations=10,
    maxsize=20,
    unary_operators=["exp", "log"],
    binary_operators=["+", "*", "-", "/"],
)

model.fit(X, y)
print("Extracted Policy Equation:", model.latex())  # Get interpretable equation
