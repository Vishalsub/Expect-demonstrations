import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

class LOKIGAgent:
    def __init__(self, policy_net, critic_net, lr=3e-4, gamma=0.99):
        self.policy_net = policy_net
        self.critic_net = critic_net
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = deque(maxlen=1000)  # Buffer to store state, action, reward, done

    def select_action(self, state, hidden_policy):
        """Select an action using the policy network."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        with torch.no_grad():
            action, hidden_policy = self.policy_net(state, hidden_policy)
        return action.numpy()[0], hidden_policy

    def store_transition(self, state, action, reward, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, done))

    def update_loki_g(self):
        """Update the policy and critic using Advantage Actor-Critic (A2C) style update."""
        if len(self.buffer) < 200:
            print("Not enough transitions in buffer. Skipping update.")
            return 0.0, 0.0

        # Unpack the buffer and convert to tensors
        states, actions, rewards, dones = zip(*self.buffer)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Add a time dimension for RNN input if needed
        if states.dim() == 2:
            states = states.unsqueeze(1)  # Shape: (batch_size, 1, state_dim)

        # Compute discounted returns
        returns = self._compute_discounted_returns(rewards, dones)

        # Ensure tensors require gradients (for debugging)
        print("Before Forward Pass:")
        print(f"states.requires_grad: {states.requires_grad}")
        print(f"actions.requires_grad: {actions.requires_grad}")

        self.optimizer_policy.zero_grad()
        self.optimizer_critic.zero_grad()

        # Forward pass through the critic network to get predicted values
        pred_values, _ = self.critic_net(states)

        # Compute the advantage: returns - predicted values
        advantage = returns - pred_values.squeeze()

        # Ensure advantage is connected to the computation graph
        print("After Advantage Calculation:")
        print(f"advantage.requires_grad: {advantage.requires_grad}")

        # Compute policy loss (broadcast advantage to match actions' shape)
        policy_loss = -(advantage.unsqueeze(1) * actions).mean()

        # Ensure policy_loss requires gradients
        print(f"policy_loss.requires_grad: {policy_loss.requires_grad}")

        # Compute critic loss (mean squared error)
        value_loss = nn.MSELoss()(pred_values.squeeze(), returns)

        # Backpropagation with gradient clipping
        policy_loss.backward(retain_graph=True)
        value_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5)

        self.optimizer_policy.step()
        self.optimizer_critic.step()

        # Clear the buffer after update
        self.buffer.clear()

        print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")
        return policy_loss.item(), value_loss.item()

    def _compute_discounted_returns(self, rewards, dones):
        """Compute discounted returns for each step in the trajectory."""
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)  # Discounted sum of rewards
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)
