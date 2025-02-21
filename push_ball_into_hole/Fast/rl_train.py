import torch
from rnn_model import RNNPolicy, RNNCritic
from loki_agent import LOKIGAgent
import gymnasium as gym


gym.register(
        id='PushingBall',
        entry_point='push:PushingBallEnv', 
        max_episode_steps=100,
    )

# Initialize the environment
env = gym.make('PushingBall', render_mode='human')
# Initialize networks and agent
policy_net = RNNPolicy(obs_dim=25, action_dim=4)
critic_net = RNNCritic(obs_dim=25)
agent = LOKIGAgent(policy_net, critic_net)

# Load pre-trained policy from Behavior Cloning
policy_net.load_state_dict(torch.load("bc_policy.pth"))

# Training loop
for epoch in range(500):
    state, _ = env.reset()
    hidden_policy = None
    total_reward = 0

    for step in range(1000):
        action, hidden_policy = agent.select_action(state['observation'], hidden_policy)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state['observation'], action, reward, done)
        state = next_state
        total_reward += reward
        if done:
            break

    policy_loss, value_loss = agent.update_loki_g()
    print(f"Epoch {epoch + 1}, Total Reward: {total_reward:.2f}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
