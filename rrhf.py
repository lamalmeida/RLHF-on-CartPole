import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
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

class RewardModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_size * 2 + action_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action, next_state):
        action_one_hot = torch.zeros(action.size(0), action_size).to(state.device)
        action_one_hot.scatter_(1, action.unsqueeze(1), 1)
        x = torch.cat([state, action_one_hot, next_state], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) 

def human_rank(state, action, next_state):
    pole_angle = abs(next_state[2])  
    base_score = math.exp(-pole_angle / 0.1)  
    noise_magnitude = pole_angle / 10
    noise = random.gauss(0, noise_magnitude) 
    noisy_score = base_score + noise
    return min(1.0, max(0.0, noisy_score))

def train_reward_model(trajectories, reward_model, optimizer, criterion):
    states = torch.stack([torch.FloatTensor(tr[0]) for tr in trajectories]).to(device)
    actions = torch.tensor([tr[1] for tr in trajectories], dtype=torch.long).to(device)
    next_states = torch.stack([torch.FloatTensor(tr[2]) for tr in trajectories]).to(device)
    scores = torch.tensor([tr[3] for tr in trajectories], dtype=torch.float32).unsqueeze(1).to(device)
    
    optimizer.zero_grad()
    predicted_scores = reward_model(states, actions, next_states)
    loss = criterion(predicted_scores, scores)
    loss.backward()
    optimizer.step()

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = QNetwork(state_size, action_size).to(device)
reward_model = RewardModel(state_size, action_size).to(device)
target_network = QNetwork(state_size, action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

q_optimizer = optim.Adam(q_network.parameters(), lr=0.001)
reward_optimizer = optim.Adam(reward_model.parameters(), lr=0.001)
q_criterion = nn.MSELoss()
reward_criterion = nn.MSELoss()

episodes = 5000
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
batch_size = 64
memory = deque(maxlen=10000)
update_target_every_steps = 1000
step_count = 0

def choose_action(state, epsilon):
    if random.random() <= epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_network(state)
        return torch.argmax(q_values, dim=1).item()

def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states = torch.stack([torch.FloatTensor(tr[0]) for tr in minibatch]).to(device)
    actions = torch.tensor([tr[1] for tr in minibatch], dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor([tr[2] for tr in minibatch], dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.stack([torch.FloatTensor(tr[3]) for tr in minibatch]).to(device)
    dones = torch.tensor([tr[4] for tr in minibatch], dtype=torch.float32).unsqueeze(1).to(device)
    
    current_q = q_network(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target_network(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (gamma * max_next_q * (1 - dones))
    
    loss = q_criterion(current_q, target_q)
    
    q_optimizer.zero_grad()
    loss.backward()
    q_optimizer.step()

scores = []
moving_avg_scores = []
window = 50
episode_lengths = []
moving_avg_lengths = []
consecutive_count = 0 
import time
start = time.time()
for e in range(1, episodes + 1):
    state, _ = env.reset()
    state = np.reshape(state, [state_size])
    total_reward = 0
    trajectories = []
    total_steps = 0

    for ttime in range(500):
        action = choose_action(state, epsilon)
        next_state, _, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [state_size])
        score = human_rank(state, action, next_state)
        trajectories.append((state, action, next_state, score, done))
        
        env_reward = 1.0  
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_tensor = torch.tensor(action).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        with torch.no_grad():
            model_reward = reward_model(state_tensor, action_tensor, next_state_tensor).squeeze(1).item()
        combined_reward = 0.5 * env_reward + 0.5 * model_reward  
        
        memory.append((state, action, combined_reward, next_state, done))
        state = next_state
        total_reward += combined_reward
        total_steps += 1
        step_count += 1

        if step_count % update_target_every_steps == 0:
            target_network.load_state_dict(q_network.state_dict())

        if done:
            break

    episode_lengths.append(total_steps)
    moving_avg = np.mean(episode_lengths[-window:])
    moving_avg_lengths.append(moving_avg)
    train_reward_model(trajectories, reward_model, reward_optimizer, reward_criterion)

    replay()

    scores.append(total_reward)
    moving_avg = np.mean(scores[-window:])
    moving_avg_scores.append(moving_avg)
    
    print(f"Episode: {e}/{episodes}, Score: {total_reward:.2f}, Avg Score: {moving_avg:.2f}, Epsilon: {epsilon:.2f}")
    
    if total_steps == 500:
        consecutive_count += 1
        if consecutive_count >= 5:
            print(f"Early stopping triggered at iteration .")
            break
    else:
        consecutive_count = 0

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
end=time.time()
print(end-start)
plt.figure(figsize=(10, 6))
plt.plot(episode_lengths, label='Episode Length')
plt.plot(range(window, len(episode_lengths)), moving_avg_lengths[window:], label=f'{window}-Episode Moving Average')
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title('Episode Length and Moving Average (CartPole-v1)')
plt.grid()
plt.legend()
plt.show()

torch.save(q_network.state_dict(), 'dqn_cartpole_model.pth')
torch.save(reward_model.state_dict(), 'reward_model.pth')
