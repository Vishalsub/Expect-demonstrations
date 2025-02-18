import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class BCDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.states = torch.tensor(np.array(data['obs']), dtype=torch.float32)
        self.actions = torch.tensor(np.array(data['action']), dtype=torch.float32)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class RNNPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(RNNPolicy, self).__init__()
        self.rnn = nn.RNN(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        action = torch.tanh(self.fc(out[:, -1, :]))
        return action, hidden

def train_bc(policy_net, data_path, epochs=10, batch_size=32, learning_rate=1e-4):
    dataset = BCDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    policy_net.train()
    for epoch in range(epochs):
        total_loss = 0
        for states, actions in dataloader:
            optimizer.zero_grad()
            output, _ = policy_net(states.unsqueeze(1))
            loss = criterion(output, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"BC Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(policy_net.state_dict(), "bc_policy.pth")
    print("Behavior Cloning complete. Model saved as bc_policy.pth")
    return policy_net

# Usage example
if __name__ == "__main__":
    policy_net = RNNPolicy(obs_dim=25, action_dim=4)
    train_bc(policy_net, data_path="./demonstration_data.npz")
