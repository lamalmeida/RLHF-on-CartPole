import gymnasium as gym
import torch
import torch.nn as nn

# Define the QNetwork model
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model(state)
    
# Create the environment with rendering enabled
env = gym.make('CartPole-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load the trained policy
q_network = QNetwork(state_dim, action_dim)
q_network.load_state_dict(torch.load('aihf_policy_cartpole.pth'))
q_network.eval()

# Initialize the environment
obs, info = env.reset()
terminated, truncated = False, False

# Play the game using the trained policy
while not (terminated or truncated):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_probs = q_network(obs_tensor)
    action = torch.argmax(action_probs, dim=1).item()  # Select the action with the highest probability
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
