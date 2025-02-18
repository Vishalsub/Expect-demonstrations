import gymnasium as gym
from push import PushingBallEnv  # Your custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

# Register your custom environment
gym.register(
    id="PushingBall-v0",
    entry_point="push:PushingBallEnv",  # Replace with the correct path
    max_episode_steps=100,
)

# Initialize the environment
env = gym.make("PushingBall-v0", render_mode="human")

# Check if the environment follows the Gymnasium API
check_env(env, warn=True)

# Wrap the environment for vectorized training
vec_env = DummyVecEnv([lambda: env])



class CustomMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.rewards = []

    def _on_step(self):
        # Log policy loss, value loss, and entropy loss
        self.policy_losses.append(self.model.logger.name_to_value.get('loss/policy_loss', None))
        self.value_losses.append(self.model.logger.name_to_value.get('train/value_loss', None))
        self.entropy_losses.append(self.model.logger.name_to_value.get('loss/entropy_loss', None))
        return True

    def _on_rollout_end(self):
        # Log episode rewards
        ep_rew_mean = self.model.logger.name_to_value.get('rollout/ep_rew_mean', None)
        if ep_rew_mean is not None:
            self.rewards.append(ep_rew_mean)

# Initialize the PPO model
model = PPO(
    "MultiInputPolicy",          # MultiInputPolicy for dict observations
    vec_env,                     # The vectorized environment
    learning_rate=3e-4,          # Learning rate
    gamma=0.99,                  # Discount factor
    verbose=1,                   # Show training logs
    tensorboard_log="./ppo_push_tensorboard/"  # Directory for TensorBoard logs
)

# Train the model with custom callback
timesteps = 100_000  # Number of timesteps to train
metrics_callback = CustomMetricsCallback()
model.learn(total_timesteps=timesteps, callback=metrics_callback)

# Save the trained model
model.save("ppo_pushing_ball3")
print("Model saved as 'ppo_pushing_ball2.zip'")

# Close the environment
vec_env.close()



