import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from push import PushingBallEnv
from tqdm import tqdm  # For progress bar

# Load the trained model
model_path = "/home/srl/Fast-RL-Demo/push_ball_into_hole/ppo_pushing_ball3.zip"

try:
    model = PPO.load(model_path)
except FileNotFoundError:
    print(f"Error: The model file at {model_path} was not found.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Register and initialize the environment
try:
    gym.register(id="PushingBall-v0", entry_point="push:PushingBallEnv", max_episode_steps=100)
    env = gym.make("PushingBall-v0", render_mode="human")
except Exception as e:
    print(f"Error initializing environment: {e}")
    exit(1)

# Check observation space of the environment
print("Observation space:", env.observation_space)

# Collect observations and actions
num_episodes = 50000
observations, actions = [], []

# Initialize tqdm progress bar
with tqdm(total=num_episodes, desc="Episodes", unit="episode") as pbar:
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False

        #print(f"Episode {episode+1}/{num_episodes} started.")
        #print(f"State structure at the beginning of episode {episode + 1}: {state}")

        # Check if the observation space is a numpy array and print it
        observation = state['observation']
        #print(f"Observation shape: {observation.shape}")  # Check the shape of observation
        #print(f"Observation values: {observation}")  # Check the values of observation

        # Ensure the observation is a numpy array and in the right shape
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        # Pass the full observation (the dictionary) to the model
        action, _ = model.predict(state, deterministic=True)

        # Print the predicted action (if needed)
        # if episode % 10 == 0:  # Print every 10th episode
        #     print(f"Predicted Action: {action}")

        # Take the step in the environment
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store the observation and action
        observations.append(observation)
        actions.append(action)

        # Update state for next iteration
        state = next_state

        #print(f"Episode {episode+1} finished.\n")
        pbar.update(1)  # Update the progress bar

# Convert to NumPy arrays
observations = np.array(observations)
actions = np.array(actions)

# Save the collected data
np.savez("collected_policy_data.npz", observations=observations, actions=actions)
print("Data collected and saved successfully!")
