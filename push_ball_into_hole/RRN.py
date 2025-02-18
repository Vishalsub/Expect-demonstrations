import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# Register the custom Push environment
gym.register(
    id='PushingBall',
    entry_point='push:PushingBallEnv', 
    max_episode_steps=100,
)

# Initialize the environment
env = gym.make('PushingBall', render_mode='human')


# RNN-based Policy Network
class RNNPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(RNNPolicy, self).__init__()
        self.rnn = nn.RNN(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        action = torch.tanh(self.fc(out[:, -1, :]))  
        return action, hidden


# RNN-based Critic Network
class RNNCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(RNNCritic, self).__init__()
        self.rnn = nn.RNN(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        if out.dim() == 3:  
            value = self.fc(out[:, -1, :])
        else: 
            value = self.fc(out)
        return value, hidden



class LOKIGAgent:
    def __init__(self, policy_net, critic_net, lr=3e-4, gamma=0.99):
        self.policy_net = policy_net
        self.critic_net = critic_net
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = deque(maxlen=1000)
    
    def select_action(self, state, hidden_policy):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action, hidden_policy = self.policy_net(state, hidden_policy)
        return action.numpy()[0], hidden_policy

    def store_transition(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))

    def train_with_imitation(self, expert_policy, epochs=100):
        """Imitation learning with KL divergence as a surrogate loss."""
        loss_fn = nn.MSELoss() 
        for epoch in range(epochs):
            self.optimizer_policy.zero_grad()
            expert_actions, _ = expert_policy(torch.randn(100, 10, 25))
            pred_actions, _ = self.policy_net(torch.randn(100, 10, 25))
            loss = loss_fn(pred_actions, expert_actions)
            loss.backward()
            self.optimizer_policy.step()
            print(f"IL Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def update_loki_g(self):
        if len(self.buffer) < 200:
            return 0.0, 0.0

        states, actions, rewards, dones = zip(*self.buffer)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        if states.dim() == 2:
            states = states.unsqueeze(1)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, requires_grad=True)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute discounted returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        self.optimizer_policy.zero_grad()
        self.optimizer_critic.zero_grad()

        pred_values, _ = self.critic_net(states)
        advantage = returns - pred_values.squeeze()  # Do not detach advantage

        # Compute policy loss and value loss
        policy_loss = -(advantage.unsqueeze(1) * actions).mean()
        value_loss = advantage.pow(2).mean()

        # Backpropagation with gradient clipping
        policy_loss.backward(retain_graph=True)  # Ensure gradients are tracked
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5)
        self.optimizer_policy.step()
        self.optimizer_critic.step()

        return policy_loss.item(), value_loss.item()



    def sample_k(self, Nm, NM):
        """Randomly sample K from range [Nm, NM]."""
        return np.random.randint(Nm, NM + 1)


# Instantiate networks and agent
obs_dim = 25
action_dim = 4
policy_net = RNNPolicy(obs_dim, action_dim)
critic_net = RNNCritic(obs_dim)
agent = LOKIGAgent(policy_net, critic_net)

# Phase 1: Imitation Learning
Nm, NM = 5, 15
K = agent.sample_k(Nm, NM)
print(f"Randomly selected K = {K}")
expert_policy = RNNPolicy(obs_dim, action_dim)  # Placeholder for expert policy
agent.train_with_imitation(expert_policy, epochs=K)

# Phase 2: Reinforcement Learning
num_rl_epochs = 500
rewards_per_episode = []
policy_losses = []
value_losses = []

for epoch in range(num_rl_epochs):
    state, info = env.reset()
    hidden_policy = None
    episode_rewards = 0

    for step in range(1000):
        action, hidden_policy = agent.select_action(state['observation'], hidden_policy)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.store_transition(state['observation'], action, reward, done)
        state = next_state
        episode_rewards += reward
        if done:
            break

    # Update LOKI-G and record losses
    policy_loss, value_loss = agent.update_loki_g()
    policy_losses.append(policy_loss)
    value_losses.append(value_loss)
    rewards_per_episode.append(episode_rewards)
    print(f"RL Epoch {epoch + 1}/{num_rl_epochs}, Total Reward: {episode_rewards}")

env.close()

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(policy_losses, label="Policy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Policy Loss during Training")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(value_losses, label="Value Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Value Loss during Training")
plt.legend()
plt.grid(True)
plt.show()
