import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from PPO_network import PushNetwork, PPOAgent
import os 

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
learning_rate = 1e-5
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 1.0
initial_entropy_coef = 0.02
final_entropy_coef = 0.0005
decay_rate = 0.999
value_loss_coef = 0.5
max_grad_norm = 1.0
num_epochs = 15000
num_steps_per_update = 512
mini_batch_size = 64
ppo_epochs = 10
max_steps_per_episode = 1001
reward_scaling = 0.001

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

def print_training_log(episode, step, policy_loss, value_loss, distance_to_goal, raw_reward, final_reward):
    print("\n+----------------------+----------------------+")
    print(f"| {'Metric':<20} | {'Value':<20} |")
    print("+----------------------+----------------------+")
    print(f"| {'Episode':<20} | {episode:<20} |")
    print(f"| {'Step':<20} | {step:<20} |")
    print(f"| {'Policy Loss':<20} | {policy_loss:<20.6f} |")
    print(f"| {'Value Loss':<20} | {value_loss:<20.6f} |")
    print(f"| {'Distance to Goal':<20} | {distance_to_goal:<20.4f} |")
    print(f"| {'Raw Reward':<20} | {raw_reward:<20.4f} |")
    print(f"| {'Final Reward':<20} | {final_reward:<20.4f} |")
    print("+----------------------+----------------------+\n")




while episode < num_epochs:
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

        # Reward function
        distance_to_goal = np.linalg.norm(desired_goal - achieved_goal)
        reward = -distance_to_goal  
        if distance_to_goal < 0.05:
            reward += 50  
        if distance_to_goal < 0.1:
            reward += 10
        reward += 5 * (1 - distance_to_goal)

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

            # Normalize advantages (inside PPO update step)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages *= reward_scaling

            # Adaptive entropy decay
            entropy_coef = max(final_entropy_coef, initial_entropy_coef * (decay_rate ** episode))

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

                    # Compute loss and update PPO
                    total_loss, policy_loss, value_loss = agent.compute_loss(
                        batch_data, gamma, gae_lambda, clip_epsilon, value_loss_coef, entropy_coef
                    )
                    agent.update(total_loss, max_grad_norm)

                    # Store losses
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    total_losses.append(total_loss.item())

                    last_idx = -1  # Get the most recent data point in the buffer
                    last_distance = np.linalg.norm(buffer.desired_goals[last_idx] - buffer.achieved_goals[last_idx])
                    last_raw_reward = buffer.rewards[last_idx] / reward_scaling
                    last_final_reward = buffer.rewards[last_idx]
   

                    # CALL THE PRINT FUNCTION HERE WITH CORRECT VALUES
                    print_training_log(
                        episode=episode,
                        step=step,
                        policy_loss=policy_loss.item(),
                        value_loss=value_loss.item(),
                        distance_to_goal=last_distance,   
                        raw_reward=last_raw_reward,       
                        final_reward=last_final_reward    
                    )
                    
                    print(f"[Step {step}] | Reward: {reward:.4f}")

            # Clear buffer
            buffer.clear()

        if done:
            break

    episode += 1
    rewards_per_episode.append(episode_rewards)
    print(f"Episode {episode} completed. Total Reward: {episode_rewards}")

torch.save(agent.network.state_dict(), "final_optimized_policy_feb-1000.pth")
print("Policy saved as 'final_optimized_policy.pth'")

env.close()

# Define the folder where you want to save the plots
save_folder = "training_plots_setup_ppo_1"
os.makedirs(save_folder, exist_ok=True)  



# Plot training results

# Plot rewards per episode# Plot rewards per episode and save it
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_folder, "total_reward_per_episode.png"))  
plt.close()  

# Plot policy loss and save it
plt.figure(figsize=(10, 5))
plt.plot(policy_losses, label="Policy Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Policy Loss during Training")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_folder, "policy_loss.png"))
plt.close()

# Plot value loss and save it
plt.figure(figsize=(10, 5))
plt.plot(value_losses, label="Value Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Value Loss during Training")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_folder, "value_loss.png"))
plt.close()

# Plot total loss and save it
plt.figure(figsize=(10, 5))
plt.plot(total_losses, label="Total Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.title("Total Loss during Training")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_folder, "total_loss.png"))
plt.close()

print(f"Plots saved in the folder: {save_folder}")
