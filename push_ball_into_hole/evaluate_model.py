import gymnasium as gym
from push import PushingBallEnv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PPO_network import PushNetwork, PPOAgent

# Register the custom Push environment
gym.register(
    id='PushingBall',
    entry_point='push:PushingBallEnv',
    max_episode_steps=1000,
)

# Initialize the environment
env = gym.make('PushingBall', render_mode='human')

# Dimensions based on the environment's observation and action spaces
obs_dim = 25
goal_dim = 3
action_dim = 4

# Re-create the network and agent
network = PushNetwork(obs_dim, goal_dim, action_dim)
agent = PPOAgent(network)

# Load the saved policy
policy_path = "trained_policy_densenonrandom_1000.pth"
agent.network.load_state_dict(torch.load(policy_path))
print("Policy loaded from 'trained_policy_densenonrandom_1000.pth'")

# Testing the policy
num_test_episodes = 100  # Number of episodes to test
rewards_per_episode = []
goal_reached = []

for episode in range(num_test_episodes):
    state, info = env.reset()
    observation = state['observation']
    achieved_goal = state['achieved_goal']
    desired_goal = state['desired_goal']
    
    episode_rewards = 0  # Total rewards for the current episode
    reached_goal = False  # Track if the goal was reached

    while True:
        # Select action using the trained policy
        action, _, _ = agent.select_action(observation, achieved_goal, desired_goal)
        
        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update the state
        observation = next_state['observation']
        achieved_goal = next_state['achieved_goal']
        desired_goal = next_state['desired_goal']
        
        # Accumulate rewards
        episode_rewards += reward

        # Check if the goal was reached
        if info.get('is_success', False):  
            reached_goal = True
        
        if done:
            break
    
    # Save results
    rewards_per_episode.append(episode_rewards)
    goal_reached.append(reached_goal)
    print(f"Test Episode {episode + 1} completed. Total Reward: {episode_rewards}, Goal Reached: {reached_goal}")

# Close the environment
env.close()

# Visualization
# Plot rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label='Reward per Episode', marker='o')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Test Episode')
plt.grid()
plt.legend()
plt.show()

# Plot success rate
success_rate = np.cumsum(goal_reached) / np.arange(1, len(goal_reached) + 1)
plt.figure(figsize=(10, 5))
plt.plot(success_rate, label='Cumulative Success Rate', marker='o', color='green')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title('Cumulative Success Rate over Test Episodes')
plt.grid()
plt.legend()
plt.show()

