import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Define the Q-Network
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

# Initialize environment and parameters
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()

episodes = 1000
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory = deque(maxlen=2000)
update_target_every = 5  

def choose_action(state, epsilon):
    if random.random() <= epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state)
        return np.argmax(q_values.numpy())

def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states, targets_f = [], []
    for state, action, reward, next_state, done in minibatch:
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        target = reward
        if not done:
            with torch.no_grad():
                target += gamma * torch.max(target_network(next_state))
        target_f = q_network(state)
        target_f = target_f.clone()
        target_f[action] = target
        states.append(state)
        targets_f.append(target_f)
    states = torch.stack(states)
    targets_f = torch.stack(targets_f)
    optimizer.zero_grad()
    outputs = q_network(states)
    loss = criterion(outputs, targets_f)
    loss.backward()
    optimizer.step()

scores = []
for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [state_size])
    total_reward = 0
    for time in range(500):
        action = choose_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        if done:
            break
        replay()
    scores.append(total_reward)
    print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.2f}")
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    if e % update_target_every == 0:
        target_network.load_state_dict(q_network.state_dict())

# Visualization
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN Training Performance on CartPole-v1')
plt.show()

# Save the trained model
torch.save(q_network.state_dict(), 'dqn_cartpole_model.pth')
