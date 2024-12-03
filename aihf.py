import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reward Model
class RewardModel(nn.Module):
    def __init__(self, state_dim):
        super(RewardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.model(state)

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
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

# Generate a trajectory
def generate_trajectory(env, policy):
    state = env.reset()
    state = np.array(state[0] if isinstance(state, tuple) else state)
    trajectory = []
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_probs = policy(state_tensor)
        action = Categorical(action_probs).sample().item()
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((state, action))
        state = np.array(next_state[0] if isinstance(next_state, tuple) else next_state)
    return trajectory

# Compute the cumulative reward of a trajectory
def compute_trajectory_reward(trajectory, reward_model):
    reward_sum = 0
    for state, _ in trajectory:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        reward_sum += reward_model(state_tensor)
    return reward_sum

average_lengths = []

# Train AIHF
def train_aihf(env, policy, reward_model, num_iterations=500, demo_trajectories=5, preference_pairs=10):
    optimizer_policy = optim.Adam(policy.parameters(), lr=1e-3)
    optimizer_reward = optim.Adam(reward_model.parameters(), lr=1e-4)
    gamma = 0.99

    for iteration in range(num_iterations):
        # Demonstration data
        demonstrations = [generate_trajectory(env, policy) for _ in range(demo_trajectories)]
        
        avg_length = np.mean([len(traj) for traj in demonstrations])
        average_lengths.append(avg_length)

        # Preferences data
        preferences = []
        for _ in range(preference_pairs):
            traj1, traj2 = generate_trajectory(env, policy), generate_trajectory(env, policy)
            reward1, reward2 = compute_trajectory_reward(traj1, reward_model), compute_trajectory_reward(traj2, reward_model)
            label = torch.tensor([[1.0 if reward1 > reward2 else 0.0]], dtype=torch.float32, device=device)
            preferences.append((traj1, traj2, label))

        # Reward Model Training
        reward_model.train()
        for traj1, traj2, label in preferences:
            optimizer_reward.zero_grad()
            reward1 = compute_trajectory_reward(traj1, reward_model)
            reward2 = compute_trajectory_reward(traj2, reward_model)
            preference_prob = torch.sigmoid(reward1 - reward2)
            loss = nn.BCELoss()(preference_prob, label)
            loss.backward()
            optimizer_reward.step()

        # Policy Training
        policy.train()
        total_steps = 0
        for trajectory in demonstrations:
            optimizer_policy.zero_grad()
            discounted_rewards, log_probs = [], []
            cumulative_reward = 0
            for state, action in reversed(trajectory):
                cumulative_reward = reward_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).item() + gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
            for state, action in trajectory:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action_probs = policy(state_tensor)
                log_probs.append(torch.log(action_probs.squeeze(0)[action]))
            policy_loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
            policy_loss.backward()
            optimizer_policy.step()
            total_steps += len(trajectory)

        print(f"Iteration {iteration + 1}/{num_iterations} completed. Ploicy Loss: {total_steps:.0f}")

# Main Function
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    reward_model = RewardModel(state_dim).to(device)

    train_aihf(env, policy, reward_model)

    torch.save(policy.state_dict(), "aihf_policy_cartpole.pth")

    moving_avg_lengths = np.convolve(
        average_lengths, np.ones(5) / 5, mode="valid"
    )
    plt.figure(figsize=(10, 6))
    plt.plot(average_lengths, label="Average Episode Length")
    plt.plot(range(5 - 1, len(average_lengths)), moving_avg_lengths, label=f"5-Episode Moving Average")
    plt.xlabel("Iteration")
    plt.ylabel("Average Length of Demo Trajectories")
    plt.title("Average Episode Lengths During Training")
    plt.legend()
    plt.grid()
    plt.show()