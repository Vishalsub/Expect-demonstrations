import gymnasium as gym
import numpy as np
import torch

class ExpertPolicy(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ExpertPolicy, self).__init__()
        self.fc = torch.nn.Linear(obs_dim, action_dim)
    
    def forward(self, x):
        return torch.tanh(self.fc(x))

def collect_demonstrations(env, expert_policy, num_episodes=20, save_path="./demonstration_data.npz"):
    data = {"obs": [], "action": []}

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state['observation'], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = expert_policy(state_tensor).numpy().squeeze()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            data["obs"].append(state["observation"])
            data["action"].append(action)
            state = next_state

    np.savez(save_path, obs=data["obs"], action=data["action"])
    print(f"Demonstration data saved at {save_path}")

# Usage example
if __name__ == "__main__":
    # Register the custom Push environment
    gym.register(
        id='PushingBall',
        entry_point='push:PushingBallEnv', 
        max_episode_steps=100,
    )

    # Initialize the environment
    env = gym.make('PushingBall', render_mode='human')
    expert_policy = ExpertPolicy(obs_dim=25, action_dim=4)  
    collect_demonstrations(env, expert_policy)
