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

# Hyperparameters
policy_learning_rate = 3e-5
value_learning_rate = 1e-5 
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
initial_entropy_coef = 0.01
final_entropy_coef = 0.0005
decay_rate = 0.999  
value_loss_coef = 0.5  
max_grad_norm = 0.5
num_epochs = 1000
num_steps_per_update = 200
mini_batch_size = 64
ppo_epochs = 10
max_steps_per_episode = 1001
reward_scaling = 0.1

#  Reward Normalization
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

# Metrics for plotting
rewards_per_episode = []
policy_losses = []
value_losses = []
total_losses = []

# Training loop
total_steps = 0
episode = 0

while episode < 1000:  
    state, info = env.reset()
    observation = state['observation']
    achieved_goal = state['achieved_goal']
    desired_goal = state['desired_goal']

    episode_rewards = 0  

    for step in range(max_steps_per_episode):
        with torch.no_grad():
            action, log_prob, value = agent.select_action(observation, achieved_goal, desired_goal)
            action = np.tanh(action)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        #  Normalize & Clip Reward
        distance_to_goal = np.linalg.norm(desired_goal - achieved_goal)
        reward = -distance_to_goal  
        if distance_to_goal < 0.05:
            reward += 10  
        reward += 5 * (1 - distance_to_goal)

        reward = reward_normalizer.normalize(reward)  # Normalize reward
        reward = np.clip(reward, -10, 10)  

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
            advantages *= reward_scaling

            #  Logarithmic Entropy Decay
            entropy_coef = max(final_entropy_coef, initial_entropy_coef * np.exp(-0.0001 * episode))

            # Optimize policy for ppo_epochs
            for _ in range(ppo_epochs):
                indices = np.arange(len(buffer.rewards))
                np.random.shuffle(indices)
                for start in range(0, len(buffer.rewards), mini_batch_size):
                    end = start + mini_batch_size
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

                    #  Use Huber Loss Instead of MSE for Stability
                    total_loss, policy_loss, value_loss = agent.compute_loss(
                        batch_data, gamma, gae_lambda, clip_epsilon, value_loss_coef, entropy_coef
                    )

                    # Track gradient norm
                    agent.optimizer.zero_grad()
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_grad_norm)
                    agent.optimizer.step()

                    # Store losses
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    total_losses.append(total_loss.item())

                    # Data (use your actual variables here)
                    print("+------------------+------------------+")
                    print(f"| {'Episode':<16} | {episode:<16} |")
                    print(f"| {'Grad Norm':<16} | {grad_norm:<16.3f} |")
                    print(f"| {'Policy Loss':<16} | {policy_loss.item():<16.4f} |")
                    print(f"| {'Value Loss':<16} | {value_loss.item():<16.4f} |")
                    print(f"| {'Step':<16} | {step:<16} |")
                    print(f"| {'Distance':<16} | {distance_to_goal:<16.4f} |")
                    print(f"| {'Raw Reward':<16} | {-distance_to_goal:<16.4f} |")
                    print(f"| {'Final Reward':<16} | {reward:<16.4f} |")
                    print("+------------------+------------------+")


            # Clear buffer
            buffer.clear()

        if done:
            break

                # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 🟢 Normalize & Clip Reward
        raw_reward = reward  # Store raw reward before normalization
        reward = reward_normalizer.normalize(reward)
        reward = np.clip(reward, -10, 10)

        # 🟢 Print reward details every 10 steps
        if step % 10 == 0:
            print(f"[Step {step}] Raw Reward: {raw_reward:.4f} | Normalized: {reward:.4f} | Distance: {np.linalg.norm(desired_goal - achieved_goal):.4f}")

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

            # 🟢 Track entropy decay
            entropy_coef = max(final_entropy_coef, entropy_coef * np.exp(-0.00001 * episode))
            print(f"[Episode {episode}] Updated Entropy Coef: {entropy_coef:.6f}")

            # PPO Optimization
            for _ in range(ppo_epochs):
                agent.optimizer.zero_grad()
                total_loss, policy_loss, value_loss = agent.compute_loss(
                    batch_data, gamma, gae_lambda, clip_epsilon, value_loss_coef, entropy_coef
                )

                # Apply gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.network.parameters(), max_grad_norm)
                agent.optimizer.step()

                # 🟢 Print gradient norm and loss values every update
                print(f"[Update] Grad Norm: {grad_norm:.3f} | Policy Loss: {policy_loss:.6f} | Value Loss: {value_loss:.6f}")

                # Store losses for logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                total_losses.append(total_loss.item())

            # 🟢 Monitor Value-to-Policy Loss Ratio
            if len(value_losses) > 10:
                loss_ratio = np.mean(value_losses[-10:]) / (np.mean(policy_losses[-10:]) + 1e-8)
                print(f"[Loss Ratio] Value/Policy Loss Ratio: {loss_ratio:.2f}")

        if done:
            break

    episode += 1
    rewards_per_episode.append(episode_rewards)

    # # Monitor Value-to-Policy Loss Ratio
    # if len(value_losses) > 10:
    #     print(f"Value Loss Ratio: {np.mean(value_losses[-10:]) / (np.mean(policy_losses[-10:]) + 1e-8):.2f}")

    # # Save model every 1,000 episodes
    # if episode % 1000 == 0:
    #     torch.save(agent.network.state_dict(), f"policy_ep_{episode}.pth")
    #     print(f"✅ Saved model at episode {episode}")

env.close()
# Plot training results

# Plot rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()

# Plot policy loss
plt.figure(figsize=(10, 5))
plt.plot(policy_losses, label="Policy Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Policy Loss during Training")
plt.legend()
plt.grid(True)
plt.show()

# Plot value loss
plt.figure(figsize=(10, 5))
plt.plot(value_losses, label="Value Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Value Loss during Training")
plt.legend()
plt.grid(True)
plt.show()

# Plot total loss
plt.figure(figsize=(10, 5))
plt.plot(total_losses, label="Total Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Total Loss during Training")
plt.legend()
plt.grid(True)
plt.show()
