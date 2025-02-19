import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from PPO_network import PushNetwork, PPOAgent

# Register the custom Push environment
gym.register(
    id='PushingBall',
    entry_point='push:PushingBallEnv', 
    max_episode_steps=100,
)

# Initialize environment
env = gym.make('PushingBall', render_mode='human')

# Dimensions based on our environment's observation and action spaces
obs_dim = 25      
goal_dim = 3       
action_dim = 4     

# Creating an instance of network and PPO agent
network = PushNetwork(obs_dim, goal_dim, action_dim)
agent = PPOAgent(network)

# Hyperparameters (Optimized)
policy_learning_rate = 3e-5
value_learning_rate = 5e-6  # Reduced for critic stability
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
final_entropy_coef = 0.0005  # Minimum entropy coefficient
value_loss_coef = 0.3  # Reduced from 0.5 to prevent critic domination
max_grad_norm = 1.0  # Gradient Clipping
num_epochs = 1000
num_steps_per_update = 200
mini_batch_size = 64
ppo_epochs = 10
max_steps_per_episode = 1001

# 🟢 Reward Normalization
class RewardNormalizer:
    def __init__(self, alpha=0.99):
        self.mean = 0
        self.var = 1
        self.alpha = alpha

    def normalize(self, reward):
        self.mean = self.alpha * self.mean + (1 - self.alpha) * reward
        self.var = self.alpha * self.var + (1 - self.alpha) * (reward - self.mean) ** 2
        return reward / (np.sqrt(self.var) + 1e-8)

reward_normalizer = RewardNormalizer()

# Rollout buffer
class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.achieved_goals = []
        self.desired_goals = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.observations.clear()
        self.achieved_goals.clear()
        self.desired_goals.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

buffer = RolloutBuffer()

# Lists to store metrics for plotting
rewards_per_episode = []
policy_losses = []
value_losses = []
total_losses = []  

# Training loop
total_steps = 0
episode = 0

while episode < num_epochs:
    state, info = env.reset()
    observation = state['observation']
    achieved_goal = state['achieved_goal']
    desired_goal = state['desired_goal']

    episode_rewards = 0  # To track episode rewards

    for step in range(max_steps_per_episode):
        # Select action using PPOAgent
        action, log_prob, value = agent.select_action(observation, achieved_goal, desired_goal)

        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 🟢 Normalize & Clip Reward
        reward = reward_normalizer.normalize(reward)
        reward = np.clip(reward, -10, 10)

        # 🟢 Print reward for debugging
        print(f"Step {step} | Raw Reward: {reward:.4f} | Normalized: {reward:.4f}")

        # Store transition in buffer
        buffer.observations.append(observation)
        buffer.achieved_goals.append(achieved_goal)
        buffer.desired_goals.append(desired_goal)
        buffer.actions.append(action)
        buffer.log_probs.append(log_prob)
        buffer.values.append(value)
        buffer.rewards.append(reward)
        buffer.dones.append(done)

        episode_rewards += reward

        # Move to next state
        observation = next_state['observation']
        achieved_goal = next_state['achieved_goal']
        desired_goal = next_state['desired_goal']

        total_steps += 1

        # PPO Update Step
        if total_steps % num_steps_per_update == 0:
            # Compute returns and advantages
            returns = []
            advantages = []
            G = 0
            adv = 0
            last_value = 0

            if not done:
                _, _, last_value = agent.select_action(observation, achieved_goal, desired_goal)
            else:
                last_value = 0

            for i in reversed(range(len(buffer.rewards))):
                mask = 1.0 - buffer.dones[i]
                G = buffer.rewards[i] + gamma * G * mask
                delta = buffer.rewards[i] + gamma * last_value * mask - buffer.values[i]
                adv = delta + gamma * gae_lambda * adv * mask
                returns.insert(0, G)
                advantages.insert(0, adv)
                last_value = buffer.values[i]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Logarithmic Entropy Decay
            entropy_coef = max(final_entropy_coef, entropy_coef * np.exp(-0.00001 * episode))

            # PPO Optimization
            for _ in range(ppo_epochs):
                agent.update(total_loss, max_grad_norm)

        if done:
            break

    episode += 1
    rewards_per_episode.append(episode_rewards)
    print(f"Episode {episode} completed. Total Reward: {episode_rewards}")

# Save the trained policy
torch.save(agent.network.state_dict(), "optimized_policy.pth")
print("Policy saved as 'optimized_policy.pth'")

env.close()
