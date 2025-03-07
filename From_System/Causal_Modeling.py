import numpy as np
import pandas as pd
from dowhy import CausalModel

# Load collected policy data
data = np.load("collected_policy_data.npz")
X = data["observations"]
y = data["actions"]

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f"state_{i}" for i in range(X.shape[1])])
df["action_0"] = y[:, 0]  # Analyze the first action dimension

# Debug: Verify data structure
print("Observations Shape:", X.shape)
print("Actions Shape:", y.shape)
print("Columns in DataFrame:", df.columns)
print(df.head())

# Define causal model
model = CausalModel(
    data=df,
    treatment="state_0",
    outcome="action_0",
    common_causes=[f"state_{i}" for i in range(1, min(5, X.shape[1]))]  # Limit confounders for stability
)

# View causal graph
model.view_model()

# Identify causal effect
identified_estimand = model.identify_effect()
print("\nIdentified Causal Effect:")
print(identified_estimand)

# Estimate causal effect using linear regression
causal_estimate = model.estimate_effect(
    identified_estimand, method_name="backdoor.linear_regression"
)
print("\nCausal Effect Estimate:", causal_estimate.value)

# Placebo Test
print("\nRunning Placebo Test...")
res_placebo = model.refute_estimate(
    identified_estimand, causal_estimate, method_name="placebo_treatment_refuter",
    placebo_type="permute"
)
print(res_placebo)

# Bootstrap Test
print("\nRunning Bootstrap Test...")
res_bootstrap = model.refute_estimate(
    identified_estimand, causal_estimate, method_name="bootstrap_refuter",
    num_simulations=10
)
print(res_bootstrap)
