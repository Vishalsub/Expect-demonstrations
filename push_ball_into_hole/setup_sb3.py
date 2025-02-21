import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
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


class CustomMetricsCallback(BaseCallback):
    """
    Custom callback to log policy loss, value loss, entropy loss, and episode rewards.
    """
    def __init__(self):
        super().__init__()
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.rewards = []

    def _on_step(self) -> bool:
        """
        Logs metrics at every step.
        """
        policy_loss = self.model.logger.name_to_value.get("train/policy_loss", None)
        value_loss = self.model.logger.name_to_value.get("train/value_loss", None)
        entropy_loss = self.model.logger.name_to_value.get("train/entropy_loss", None)

        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        if entropy_loss is not None:
            self.entropy_losses.append(entropy_loss)

        return True

    def _on_rollout_end(self):
        """
        Logs average episode rewards at the end of each rollout.
        """
        ep_rew_mean = self.model.logger.name_to_value.get("rollout/ep_rew_mean", None)
        if ep_rew_mean is not None:
            self.rewards.append(ep_rew_mean)

    def save_plots(self):
        """
        Save training plots for policy loss, value loss, entropy loss, and episode rewards.
        """
        # Define plot names
        plots = {
            "Policy Loss": self.policy_losses,
            "Value Loss": self.value_losses,
            "Entropy Loss": self.entropy_losses,
            "Total Reward per Episode": self.rewards,
        }
        
        for plot_name, data in plots.items():
            if len(data) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(data, label=plot_name)
                plt.xlabel("Training Steps")
                plt.ylabel(plot_name)
                plt.title(plot_name)
                plt.legend()
                plt.grid(True)
                
                # Save plot
                plot_path = os.path.join(plot_dir, f"{plot_name.lower().replace(' ', '_')}.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved {plot_name} plot to {plot_path}")


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

# Train the model with custom callback
timesteps =  5_000_000  
metrics_callback = CustomMetricsCallback()
model.learn(total_timesteps=timesteps, callback=metrics_callback)


# Save the trained model
model_path = "ppo_training_models_sb3/ppo_pushing_ball3.zip"
model.save(model_path)
print(f"✅ Model saved as '{model_path}'")

# Save plots at the end of training
metrics_callback.save_plots()

# Close the environment
vec_env.close()
