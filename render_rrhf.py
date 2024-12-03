import gymnasium as gym
import torch
import torch.nn as nn

# Define the QNetwork model (same architecture as the training code)
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

# Load environment
env = gym.make('CartPole-v1', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Load the trained policy model
q_network = QNetwork(state_size, action_size)
q_network.load_state_dict(torch.load('dqn_cartpole_model.pth'))
q_network.eval()

# Initialize the environment
obs, info = env.reset()
terminated, truncated = False, False

# Play the game using the trained policy
while not (terminated or truncated):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(obs_tensor)
    action = torch.argmax(q_values, dim=1).item()  # Select the action with the highest Q-value
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
