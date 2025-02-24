import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from push import PushingBallEnv  # Import your custom environment

# Load trained model
model_path = "ppo_pushing_ball.zip"  # Update if needed
model = PPO.load(model_path)

# Set up the environment
env = gym.make("PushingBall-v0", render_mode=None)  # No rendering for speed

NUM_EPISODES = 1000  # Number of test episodes
success_count = 0
total_rewards = []
perturbation_test = True  # Enable random perturbations for robustness test

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        episode_reward += reward

        # If perturbation test is enabled, apply small random noise to actions
        if perturbation_test:
            action += np.random.normal(0, 0.02, size=action.shape)  # Add small noise

    total_rewards.append(episode_reward)
    if info.get("is_success", False):  # Check success metric
        success_count += 1

    print(f"Episode {episode+1}/{NUM_EPISODES}: Reward = {episode_reward}, Success = {info.get('is_success', False)}")

# Compute success rate
success_rate = success_count / NUM_EPISODES
avg_reward = np.mean(total_rewards)

print("\n==== Evaluation Results ====")
print(f"✅ Success Rate: {success_rate * 100:.2f}%")
print(f"📊 Average Reward: {avg_reward:.2f}")

# Save results for plotting
np.save("evaluation_success_rate.npy", np.array(success_rate))
np.save("evaluation_rewards.npy", np.array(total_rewards))

env.close()


import numpy as np
import matplotlib.pyplot as plt

# Load evaluation results
success_rate = np.load("evaluation_success_rate.npy")
total_rewards = np.load("evaluation_rewards.npy")

# Success rate plot
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(success_rate) / np.arange(1, len(success_rate) + 1), 'g.-', label="Cumulative Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Cumulative Success Rate over Test Episodes")
plt.legend()
plt.grid()
plt.savefig("cumulative_success_rate.png")
plt.show()

# Rewards plot
plt.figure(figsize=(8, 5))
plt.plot(total_rewards, 'b.-', label="Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Rewards per Test Episode")
plt.legend()
plt.grid()
plt.savefig("total_reward_per_episode.png")
plt.show()
