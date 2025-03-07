import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
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


### 🔹 SHAP Callback for Feature Importance Analysis
class SHAPCallback(BaseCallback):
    """
    Callback to analyze feature importance using SHAP during RL training.
    Logs and visualizes SHAP values dynamically.
    """
    def __init__(self, env, model, check_every=1000):
        super().__init__()
        self.env = env
        self.model = model
        self.check_every = check_every
        self.shap_values_history = []
        self.feature_names = None  # Will be set dynamically

    def _on_step(self) -> bool:
        """
        Run SHAP analysis every 'check_every' steps.
        """
        if self.n_calls % self.check_every == 0:
            self.run_shap_analysis()
        return True

    def run_shap_analysis(self):
        """
        Compute and log SHAP values to analyze feature importance dynamically.
        """
        print("\n🔍 Running SHAP Analysis...")
        
        # Collect a batch of state observations
        obs_samples = np.array([self.env.reset()[0]["observation"] for _ in range(100)])
        if self.feature_names is None:
            self.feature_names = [f"state_{i}" for i in range(obs_samples.shape[1])]

        # Convert to Tensor for SHAP
        obs_tensor = torch.tensor(obs_samples, dtype=torch.float32)

        # Define SHAP explainer
        def policy_fn(obs_np):
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32)
            with torch.no_grad():
                return self.model.policy.forward(obs_tensor).numpy()  # Ensure output is NumPy

        # Use the proper SHAP KernelExplainer
        explainer = shap.Explainer(policy_fn, obs_samples)
        shap_values = explainer(obs_samples)  # Call the explainer correctly

        # Store SHAP values
        self.shap_values_history.append(shap_values)

        # SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, obs_samples, feature_names=self.feature_names, show=False)
        shap_plot_path = os.path.join(plot_dir, f"shap_plot_step_{self.num_timesteps}.png")
        plt.savefig(shap_plot_path)
        plt.close()

        print(f"✅ SHAP Analysis Completed. Plot saved at {shap_plot_path}")


    def save_shap_trend(self):
        """
        Save SHAP value trends over multiple evaluations.
        """
        if not self.shap_values_history:
            print("⚠️ No SHAP values recorded. Skipping trend visualization.")
            return
        
        avg_shap_values = np.mean([np.abs(s.values) for s in self.shap_values_history], axis=0)
        feature_importance = np.mean(avg_shap_values, axis=0)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=self.feature_names, y=feature_importance)
        plt.xticks(rotation=45)
        plt.xlabel("State Features")
        plt.ylabel("Mean SHAP Value")
        plt.title("SHAP Feature Importance Over Training")
        trend_plot_path = os.path.join(plot_dir, "shap_trend.png")
        plt.savefig(trend_plot_path)
        plt.close()

        print(f"📊 SHAP Trend Plot Saved at {trend_plot_path}")


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


### 🔹 Train PPO with SHAP Callback
shap_callback = SHAPCallback(normalized_env, model, check_every=1000)

# Train the model
timesteps = 50000
model.learn(total_timesteps=timesteps, callback=[shap_callback])

# Save the trained model
model_path = "ppo_training_models_sb3/ppo_pushing_ball3.zip"
model.save(model_path)
print(f"✅ Model saved as '{model_path}'")

# Save SHAP Trend Visualization
shap_callback.save_shap_trend()

# Close the environment
vec_env.close()
