import torch
import torch.nn as nn

class RNNPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(RNNPolicy, self).__init__()
        self.rnn = nn.RNN(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        action = torch.tanh(self.fc(out[:, -1, :]))
        return action, hidden

class RNNCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(RNNCritic, self).__init__()
        self.rnn = nn.RNN(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        value = self.fc(out[:, -1, :])
        return value, hidden
