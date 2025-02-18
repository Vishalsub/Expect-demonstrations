import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layer(x)  # Residual connection

class PushNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim):
        super(PushNetwork, self).__init__()

        # Observation processing network
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Goal processing network
        self.goal_net = nn.Sequential(
            nn.Linear(goal_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Combine features
        combined_dim = 128 + 128

        # Policy network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.action_mean = nn.Linear(256, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * -2)  # Lower initial std

        # Value network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation, achieved_goal, desired_goal):
        # Process observation
        obs_features = self.obs_net(observation)

        # Process goals
        goal_input = torch.cat([achieved_goal, desired_goal], dim=-1)
        goal_features = self.goal_net(goal_input)

        # Combine features
        combined_features = torch.cat([obs_features, goal_features], dim=-1)

        # Policy network
        policy_features = self.policy_net(combined_features)
        action_mean = self.action_mean(policy_features)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Value network
        value = self.value_net(combined_features)

        return action_mean, action_std, value

class PPOAgent:
    def __init__(self, network, learning_rate=3e-4):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

    def select_action(self, observation, achieved_goal, desired_goal, deterministic=False):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        achieved_goal = torch.tensor(achieved_goal, dtype=torch.float32).unsqueeze(0).to(self.device)
        desired_goal = torch.tensor(desired_goal, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, action_std, value = self.network(observation, achieved_goal, desired_goal)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()

            action = torch.tanh(action)  # Clip actions smoothly
            action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            value = value.squeeze(-1)

        return action.cpu().numpy()[0], action_log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def update(self, total_loss, max_grad_norm):
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
        self.optimizer.step()
        
    def compute_loss(self, batch_data, gamma, gae_lambda, clip_epsilon, value_loss_coef, entropy_coef):
        """Compute PPO loss: policy loss, value loss, and entropy loss."""
        
        observations = torch.tensor(batch_data['observations'], dtype=torch.float32).to(self.device)
        achieved_goals = torch.tensor(batch_data['achieved_goals'], dtype=torch.float32).to(self.device)
        desired_goals = torch.tensor(batch_data['desired_goals'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch_data['actions'], dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(batch_data['old_log_probs'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(batch_data['returns'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(batch_data['advantages'], dtype=torch.float32).to(self.device)

        # Get action distribution and values
        action_mean, action_std, values = self.network(observations, achieved_goals, desired_goals)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        action_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1)

        values = values.squeeze(-1)

        # Compute the ratio of new and old action probabilities
        ratios = torch.exp(action_log_probs - old_log_probs)

        # Compute clipped surrogate loss
        surr1 = ratios * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss using Mean Squared Error
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy loss for exploration regularization
        entropy_loss = -dist_entropy.mean() * entropy_coef

        # Total loss
        total_loss = policy_loss + value_loss_coef * value_loss + entropy_loss

        return total_loss, policy_loss, value_loss

