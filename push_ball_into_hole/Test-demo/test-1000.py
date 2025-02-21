import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from PPO_network import PushNetwork, PPOAgent

# Register the custom Push environment
gym.register(
    id='PushingBall',
    entry_point='push:PushingBallEnv', 
    max_episode_steps=200,
)

# Initialize environment
env = gym.make('PushingBall', render_mode='human')

# Create network and PPO agent
network = PushNetwork(obs_dim=25, goal_dim=3, action_dim=4)
agent = PPOAgent(network)

# Optimized Hyperparameters
hyperparams = {
    "learning_rate": 1e-4,  # Increased LR
    "gamma": 0.99,
    "gae_lambda": 0.9,  # Slightly lower λ
    "clip_epsilon": 0.2,
    "entropy_coef": 0.005,  # Better exploration control
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_epochs": 3000,  # Increased for more training
    "num_steps_per_update": 256,  # Batch size tuning
    "mini_batch_size": 64,
    "ppo_epochs": 10,
    "max_steps_per_episode": 500,  # Lower episode length
}

# Rollout buffer for PPO
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.achieved_goals = []
        self.desired_goals = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

buffer = RolloutBuffer()

# Metrics for tracking performance
rewards_per_episode = []
success_rates = []
policy_losses = []
value_losses = []
total_losses = []

# Training loop
total_steps = 0
episode = 0

while episode < hyperparams["num_epochs"]:
    state, info = env.reset()
    observation = state['observation']
    achieved_goal = state['achieved_goal']
    desired_goal = state['desired_goal']

    episode_rewards = 0  
    success_count = 0  # Track successes

    for step in range(hyperparams["max_steps_per_episode"]):
        with torch.no_grad():
            action, log_prob, value = agent.select_action(observation, achieved_goal, desired_goal)
            action = np.tanh(action)  # Ensure valid range

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # **Improved Reward Function**
        distance_to_goal = np.linalg.norm(desired_goal - achieved_goal)
        reward = -distance_to_goal + 10 * (distance_to_goal < 0.05) + 5 * (1 - distance_to_goal)

        # Success metric
        if distance_to_goal < 0.05:
            success_count += 1

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
        if total_steps % hyperparams["num_steps_per_update"] == 0:
            returns, advantages = [], []
            G, adv, last_value = 0, 0, 0

            if not done:
                _, _, last_value = agent.select_action(observation, achieved_goal, desired_goal)
            else:
                last_value = 0

            for i in reversed(range(len(buffer.rewards))):
                mask = 1.0 - buffer.dones[i]
                G = buffer.rewards[i] + hyperparams["gamma"] * G * mask
                delta = buffer.rewards[i] + hyperparams["gamma"] * last_value * mask - buffer.values[i]
                adv = delta + hyperparams["gamma"] * hyperparams["gae_lambda"] * adv * mask
                returns.insert(0, G)
                advantages.insert(0, adv)
                last_value = buffer.values[i]

            # Convert to numpy arrays
            observations = np.array(buffer.observations)
            achieved_goals = np.array(buffer.achieved_goals)
            desired_goals = np.array(buffer.desired_goals)
            actions = np.array(buffer.actions)
            old_log_probs = np.array(buffer.log_probs)
            returns = np.array(returns)
            advantages = np.array(advantages)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Optimize policy
            for _ in range(hyperparams["ppo_epochs"]):
                indices = np.arange(len(buffer.rewards))
                np.random.shuffle(indices)
                for start in range(0, len(buffer.rewards), hyperparams["mini_batch_size"]):
                    end = start + hyperparams["mini_batch_size"]
                    mb_indices = indices[start:end]

                    batch_data = {
                        'observations': observations[mb_indices],
                        'achieved_goals': achieved_goals[mb_indices],
                        'desired_goals': desired_goals[mb_indices],
                        'actions': actions[mb_indices],
                        'old_log_probs': old_log_probs[mb_indices],
                        'returns': returns[mb_indices],
                        'advantages': advantages[mb_indices],
                    }

                    # Compute loss and update PPO
                    total_loss, policy_loss, value_loss = agent.compute_loss(
                        batch_data, hyperparams["gamma"], hyperparams["gae_lambda"],
                        hyperparams["clip_epsilon"], hyperparams["value_loss_coef"], hyperparams["entropy_coef"]
                    )
                    agent.update(total_loss, hyperparams["max_grad_norm"])

                    # Store losses
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    total_losses.append(total_loss.item())

            # Clear buffer
            buffer.clear()

        if done:
            break

    episode += 1
    rewards_per_episode.append(episode_rewards)
    success_rates.append(success_count / hyperparams["max_steps_per_episode"])

    print(f"Episode {episode} | Reward: {episode_rewards:.2f} | Success Rate: {success_rates[-1]:.4f}")

torch.save(agent.network.state_dict(), "final_optimized_policy_3000.pth")
print("Policy saved!")

env.close()

# **Plot Training Results**
plt.figure(figsize=(12, 5))
plt.plot(rewards_per_episode, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(success_rates, label="Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Success Rate Over Episodes")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(policy_losses, label="Policy Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Policy Loss During Training")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(value_losses, label="Value Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Value Loss During Training")
plt.legend()
plt.grid(True)
plt.show()
