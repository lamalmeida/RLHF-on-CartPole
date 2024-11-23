import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import math

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

# Reward Model
class RewardModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(RewardModel, self).__init__()
        self.action_size = action_size  # Store action_size
        self.fc1 = nn.Linear(state_size * 2 + action_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action, next_state):
        action_one_hot = torch.zeros(action.size(0), self.action_size).to(state.device)
        action_one_hot.scatter_(1, action.unsqueeze(1), 1)
        x = torch.cat([state, action_one_hot, next_state], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Environment setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize networks
q_network = QNetwork(state_size, action_size).to(device)
target_network = QNetwork(state_size, action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()  # Set to evaluation mode

reward_model = RewardModel(state_size, action_size).to(device)

# Optimizers
q_optimizer = optim.Adam(q_network.parameters(), lr=0.001)
reward_optimizer = optim.Adam(reward_model.parameters(), lr=0.001)

# Hyperparameters
gamma = 0.99
batch_size = 64
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
memory = deque(maxlen=10000)

# Helper functions
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        return torch.argmax(q_values, dim=1).item()

def collect_trajectories(policy, env, epsilon, steps=1000):
    trajectories = []
    state, _ = env.reset()
    for _ in range(steps):
        action = choose_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        trajectories.append((state, action, next_state, reward))
        state = next_state
        if done:
            state, _ = env.reset()
    return trajectories

def rank_trajectories(trajectories, human_rank_fn):
    ranked_data = []
    scores = []
    for state, action, next_state, _ in trajectories:
        scores.append(human_rank_fn(state, action, next_state))
    for i, traj1 in enumerate(trajectories):
        for j, traj2 in enumerate(trajectories):
            if i != j: 
                if scores[i] > scores[j]:
                    better = traj1[:3]  
                    worse = traj2[:3]
                    ranked_data.append((better, worse))
    return ranked_data

def train_reward_model_pairwise(reward_model, optimizer, ranked_data, epochs=3, batch_size=32):
    reward_model.train()
    for epoch in range(epochs):
        total_loss = 0
        # Shuffle ranked data
        random.shuffle(ranked_data)
        for i in range(0, len(ranked_data), batch_size):
            batch = ranked_data[i:i + batch_size]
            
            # Unpack batch
            states_better, actions_better, next_states_better = zip(*[b[0] for b in batch])
            states_worse, actions_worse, next_states_worse = zip(*[b[1] for b in batch])
            
            # Convert lists of NumPy arrays to single NumPy arrays
            states_better = np.array(states_better, dtype=np.float32)
            actions_better = np.array(actions_better, dtype=np.int64)
            next_states_better = np.array(next_states_better, dtype=np.float32)

            states_worse = np.array(states_worse, dtype=np.float32)
            actions_worse = np.array(actions_worse, dtype=np.int64)
            next_states_worse = np.array(next_states_worse, dtype=np.float32)

            # Convert to PyTorch tensors
            states_better = torch.tensor(states_better).to(device)
            actions_better = torch.tensor(actions_better).to(device)
            next_states_better = torch.tensor(next_states_better).to(device)

            states_worse = torch.tensor(states_worse).to(device)
            actions_worse = torch.tensor(actions_worse).to(device)
            next_states_worse = torch.tensor(next_states_worse).to(device)

            # Calculate rewards
            r_better = reward_model(states_better, actions_better, next_states_better)
            r_worse = reward_model(states_worse, actions_worse, next_states_worse)

            # Compute pairwise loss
            loss = -torch.log(torch.sigmoid(r_better - r_worse)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


# DQN Training Logic
def train_q_network_dqn(q_network, target_network, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, next_states, rewards, dones = zip(*batch)

    # Convert lists to NumPy arrays first for efficiency
    try:
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(np.stack(rewards), dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
    except ValueError as e:
        print("Error in batching data:", e)
        return

    current_q = q_network(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = target_network(next_states)
        next_q = next_q_values.max(1)[0].unsqueeze(1)

    target_q = rewards + (gamma * next_q * (1 - dones))

    loss = nn.MSELoss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"DQN Training Loss: {loss.item():.4f}")

# Human ranking function with exponential decay and noise
def human_rank(state, action, next_state):
    pole_angle = abs(next_state[2])  
    base_score = math.exp(-pole_angle / 0.1)  
    noise_magnitude = pole_angle / 10  
    noise = random.gauss(0, noise_magnitude) 
    noisy_score = base_score + noise
    return min(1.0, max(0.0, noisy_score))

# Training loop
episodes = 50000
steps_per_epoch = 500

for episode in range(episodes):
    # Collect trajectories
    trajectories = collect_trajectories(q_network, env, epsilon, steps=steps_per_epoch)

    # Add to memory
    for traj in trajectories:
        state, action, next_state, reward = traj
        done = float(traj[3] > 0)  # Adjust based on your definition of 'done'
        memory.append((
            np.array(state, dtype=np.float32),
            action,
            np.array(next_state, dtype=np.float32),
            reward,
            done
        ))

    # Rank trajectories using the human ranking function
    ranked_data = rank_trajectories(trajectories, human_rank)

    # Train the reward model
    train_reward_model_pairwise(reward_model, reward_optimizer, ranked_data)

    # Compute RM-based rewards for trajectories
    rewards = []
    for state, action, next_state, _ in trajectories:
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
        action_tensor = torch.LongTensor([action]).to(device)
        next_state_tensor = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            reward = reward_model(state_tensor, action_tensor, next_state_tensor).item()
        rewards.append(reward)

    # Train the Q-network using DQN
    train_q_network_dqn(q_network, target_network, memory, q_optimizer, batch_size, gamma)

    # Update target network periodically
    if episode % 100 == 0:
        target_network.load_state_dict(q_network.state_dict())
        print(f"Target network updated at episode {episode}")

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Print progress
    if (episode + 1) % 10 == 0:  # Adjust frequency as needed
        print(f"Episode {episode + 1}/{episodes}, RM Reward Avg: {np.mean(rewards):.4f}, Epsilon: {epsilon:.3f}")

    # Save models periodically
    if (episode + 1) % 100 == 0:
        torch.save(q_network.state_dict(), f'q_network_rmhf_episode_{episode + 1}.pth')
        torch.save(reward_model.state_dict(), f'reward_model_episode_{episode + 1}.pth')
        print(f"Models saved at episode {episode + 1}")
