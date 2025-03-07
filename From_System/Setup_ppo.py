import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from sklearn.preprocessing import StandardScaler
from push import PushingBallEnv  

# Register the custom environment
gym.register(
    id="PushingBall-v0",
    entry_point="push:PushingBallEnv",
    max_episode_steps=100,
)

# Initialize the environment
env = gym.make("PushingBall-v0", render_mode="human")

# Check if the environment follows the Gymnasium API
check_env(env, warn=True)

# Wrap the environment for vectorized training
vec_env = DummyVecEnv([lambda: env])

# Create folder to store plots
plot_dir = "ppo_training_plots"
os.makedirs(plot_dir, exist_ok=True)

# Configure TensorBoard logger
tensorboard_log_dir = "./ppo_push_tensorboard/"
logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])

# Define PPO Hyperparameters
model = PPO(
    policy="MultiInputPolicy",
    env=vec_env,
    learning_rate=5e-4,  
    gamma=0.995,  
    n_steps=1024,  
    batch_size=64,
    ent_coef=0.0025,  
    clip_range=0.2,
    gae_lambda=0.95,
    verbose=1,
    tensorboard_log=tensorboard_log_dir
)

# Set TensorBoard logger
model.set_logger(logger)

### 🔹 Feature Normalization (Preprocessing)
class FeatureNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def normalize(self, X):
        if not self.fitted:
            self.scaler.fit(X)
            self.fitted = True
        return self.scaler.transform(X)

# Initialize Normalizer
feature_normalizer = FeatureNormalizer()

### 🔹 Modify Environment Step to Normalize Observations
class NormalizedEnv(gym.Wrapper):
    """
    Wrapper to normalize environment observations dynamically.
    """
    def __init__(self, env, normalizer):
        super().__init__(env)
        self.normalizer = normalizer

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs["observation"] = self.normalizer.normalize(obs["observation"])
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["observation"] = self.normalizer.normalize(obs["observation"])
        return obs, info

# Wrap the environment with normalization
normalized_env = NormalizedEnv(env, feature_normalizer)
vec_env = DummyVecEnv([lambda: normalized_env])

# Train PPO
timesteps = 50000
model.learn(total_timesteps=timesteps)

# Save the trained model
model_path = "ppo_training_models_sb3/ppo_pushing_ball3.zip"
model.save(model_path)
print(f"✅ Model saved as '{model_path}'")

# Close the environment
vec_env.close()
