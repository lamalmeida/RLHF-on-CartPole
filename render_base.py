import gymnasium as gym
import torch
import torch.nn as nn

# Define the Policy model
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Outputs action probabilities
    
# Create the environment with rendering enabled
env = gym.make('CartPole-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load the trained policy
policy = Policy(state_dim, action_dim)
policy.load_state_dict(torch.load('cartpole_policy.pth'))
policy.eval()

# Initialize the environment
obs, info = env.reset()
terminated, truncated = False, False

# Play the game using the trained policy
while not (terminated or truncated):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_probs = policy(obs_tensor)
    action = torch.argmax(action_probs, dim=1).item()  # Select the action with the highest probability
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
