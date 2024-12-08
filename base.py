import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class RewardModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PreferenceBasedRL:
    def __init__(self, env, policy, reward_model):
        self.env = env
        self.policy = policy
        self.reward_model = reward_model
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.gamma = 0.99

    def collect_trajectories(self, n=10):
        trajectories = []
        for _ in range(n):
            state = self.env.reset()
            trajectory = []
            done = False
            while not done:
                state_tensor = torch.FloatTensor(np.array(state[0] if isinstance(state, tuple) else state)).unsqueeze(0)
                action_probs = self.policy(state_tensor)
                action_dist = Categorical(action_probs)
                action = action_dist.sample().item()
                next_state, _, terminate, truncate, _ = self.env.step(action)
                done = terminate or truncate
                trajectory.append((state, action))
                state = next_state
            trajectories.append(trajectory)
        return trajectories

    def simulate_preferences(self, trajectories):
        preferences = []
        for _ in range(len(trajectories) // 2):
            traj1, traj2 = random.sample(trajectories, 2)
            reward1 = sum(self.predict_reward(traj1))
            reward2 = sum(self.predict_reward(traj2))
            if reward1 > reward2:
                preferences.append((traj1, traj2))
            else:
                preferences.append((traj2, traj1))
        return preferences

    def predict_reward(self, trajectory):
        rewards = []
        for state, action in trajectory:
            state_tensor = torch.FloatTensor(np.array(state[0] if isinstance(state, tuple) else state)).unsqueeze(0)
            action_tensor = torch.FloatTensor([action]).unsqueeze(0)
            reward = self.reward_model(state_tensor, action_tensor).item()
            rewards.append(reward)
        return rewards

    def train_reward_model(self, preferences):
        for traj1, traj2 in preferences:
            r1 = sum(self.predict_reward(traj1))
            r2 = sum(self.predict_reward(traj2))
            r1 = torch.tensor(r1, requires_grad=True)
            r2 = torch.tensor(r2, requires_grad=True)
            prob = torch.exp(r1) / (torch.exp(r1) + torch.exp(r2))
            loss = -torch.log(prob)
            self.reward_optimizer.zero_grad()
            loss.backward()
            self.reward_optimizer.step()

    def train_policy(self, trajectories):
        returns = []
        for trajectory in trajectories:
            rewards = self.predict_reward(trajectory)
            G = 0
            discounts = []
            for r in rewards[::-1]:
                G = r + self.gamma * G
                discounts.insert(0, G)
            returns.extend(discounts)

        returns = torch.FloatTensor(returns)
        states = []
        actions = []
        for trajectory in trajectories:
            for state, action in trajectory:
                states.append(torch.FloatTensor(np.array(state[0] if isinstance(state, tuple) else state)))
                actions.append(torch.tensor(action))

        states = torch.stack(states)
        actions = torch.stack(actions)
        action_probs = self.policy(states)
        action_dists = Categorical(action_probs)
        log_probs = action_dists.log_prob(actions)
        loss = -(log_probs * returns).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def train(self, epochs=1000, traj_per_epoch=10):
        losses = []
        consecutive_count = 0 
        for epoch in range(epochs):
            trajectories = self.collect_trajectories(traj_per_epoch)
            trajectory_averages = np.mean([len(traj) for traj in trajectories]) 
            preferences = self.simulate_preferences(trajectories)
            self.train_reward_model(preferences)
            self.train_policy(trajectories)
            losses.append(np.mean([len(traj) for traj in trajectories]))
            print(f"Epoch {epoch+1}/{epochs} Loss: {losses[-1]}")
            if trajectory_averages >= 500:
                consecutive_count += 1
                if consecutive_count >= 5:
                    print(f"Early stopping triggered at epoch {epoch+1}. Saving models and plotting.")
                    break
            else:
                consecutive_count = 0
        return losses

import time
times = []
for i in range(10):
    env = gym.make('CartPole-v1')
    policy = Policy(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    reward_model = RewardModel(state_size=env.observation_space.shape[0], action_size=1)
    agent = PreferenceBasedRL(env, policy, reward_model)
    start = time.time()
    losses = agent.train()
    end = time.time()
    times.append(end-start)
print(np.mean(times))
window=50
moving_average = np.convolve(losses, np.ones(window)/window, mode='valid')
env.close()

plt.plot(losses)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, label='Episode Length')
plt.plot(range(window, len(losses) + 1), moving_average, label=f'{window}-Episode Moving Average')
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title('Episode Length and Moving Average (CartPole-v1)')
plt.grid()
plt.legend()
plt.show()

torch.save(policy.state_dict(), "cartpole_policy.pth")
torch.save(reward_model.state_dict(), "cartpole_reward_model.pth")
