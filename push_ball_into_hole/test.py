import gymnasium as gym
import torch
import numpy as np
from push import PushingBallEnv
from PPO_network import PushNetwork, PPOAgent

# Register the environment
gym.register(
    id='PushingBall',
    entry_point='push:PushingBallEnv', 
    max_episode_steps=500,
)

# Load trained PPO agent
agent = PPOAgent(PushNetwork(obs_dim=25, goal_dim=3, action_dim=4))
agent.network.load_state_dict(torch.load("final_optimized_policy.pth", weights_only=True))

# Expert policy function
def expert_policy(observation, achieved_goal, desired_goal):
    """Returns an action from the trained PPO expert model in deterministic mode."""
    with torch.no_grad():
        action, _, _ = agent.select_action(observation, achieved_goal, desired_goal, deterministic=True)
        action = np.clip(action, -1, 1)  
    return action

# Initialize environment
env = gym.make('PushingBall', render_mode='human')

# Storage for demonstrations
demonstrations = {"observations": [], "actions": [], "rewards": []}

num_demonstrations = 500

for episode in range(num_demonstrations):
    state, info = env.reset()
    observation = state['observation']
    achieved_goal = state['achieved_goal']
    desired_goal = state['desired_goal']

    episode_data = {"observations": [], "actions": [], "rewards": []}

    while True:
        action = expert_policy(observation, achieved_goal, desired_goal)

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store all actions, but later we filter based on reward
        episode_data["observations"].append(observation)
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)

        # Move to next state
        observation = next_state['observation']
        
        if done:
            break


    threshold = np.percentile(episode_data["rewards"], 75)
    for i in range(len(episode_data["rewards"])):
        if episode_data["rewards"][i] >= threshold:
            demonstrations["observations"].append(episode_data["observations"][i])
            demonstrations["actions"].append(episode_data["actions"][i])

    print(f"Episode {episode + 1}/{num_demonstrations} collected.")

# Convert lists to numpy arrays
demonstrations["observations"] = np.array(demonstrations["observations"])
demonstrations["actions"] = np.array(demonstrations["actions"])

# Save dataset
np.savez("f_high_quality_demonstrations_1.npz", **demonstrations)
print("Saved high-quality expert demonstrations to 'filtered_high_quality_demonstrations.npz'.")

env.close()
