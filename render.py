import gymnasium as gym
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# Create the environment with rendering enabled
env = gym.make('CartPole-v1', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_network = QNetwork(state_size, action_size)
q_network.load_state_dict(torch.load('q_network_rmhf_episode_5000.pth'))
q_network.eval()

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q_values = q_network(obs_tensor)
    action = torch.argmax(q_values, dim=1).item()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
