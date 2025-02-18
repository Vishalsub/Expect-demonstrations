import numpy as np
# # Load dataset


#data = np.load("high_quality_demonstrations.npz")
# data = np.load("f_high_quality_demonstrations.npz")
# observations = data["observations"]
# actions = data["actions"]

# print("Observations Shape:", observations.shape)
# print("Actions Shape:", actions.shape)

# # Print some samples
# print("Sample Observations:", observations[:5])
# print("Sample Actions:", actions[:5])

import matplotlib.pyplot as plt


#data = np.load("high_quality_demonstrations.npz")
data = np.load("f_high_quality_demonstrations.npz")
actions = data["actions"]



plt.figure(figsize=(10, 5))
plt.plot(actions[:500]) 
plt.xlabel("Timestep")
plt.ylabel("Action Value")
plt.title("Expert Policy Actions Over Time")
plt.legend(["Action 1", "Action 2", "Action 3", "Action 4"])
plt.grid(True)
plt.show()
