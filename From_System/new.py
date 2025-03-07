import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from push import PushingBallEnv  

# Load the trained model
model_path = "/Users/vishal/Expect-demonstrations/push_ball_into_hole/ppo_training_models_sb3/ppo_pushing_ball3.zip"
model = PPO.load(model_path)

# Register and initialize the environment
gym.register(id="PushingBall-v0", entry_point="push:PushingBallEnv", max_episode_steps=100)
env = gym.make("PushingBall-v0", render_mode="human")

# Collect observations and actions
num_episodes = 100
observations, actions = [], []

for _ in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(state["observation"], deterministic=True)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(state["observation"])
        actions.append(action)
        
        state = next_state

# Convert to NumPy arrays
observations = np.array(observations)
actions = np.array(actions)

# Save the collected data
np.savez("collected_policy_data.npz", observations=observations, actions=actions)
print("✅ Data collected and saved successfully!")
