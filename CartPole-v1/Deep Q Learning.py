import time
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym

# -------------------------------
# DQN network and replay memory
# -------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First layer, 128 neurons
        self.fc2 = nn.Linear(128, 128)  # Another hidden layer
        self.fc3 = nn.Linear(128, output_dim)  # Output layer (Q-values for actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output raw Q-values

# Simple replay buffer for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  # Limited size queue

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Store transition

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # Random sampling for training

    def __len__(self):
        return len(self.memory)  # Current size of memory

# -------------------------------
# DQN Agent
# -------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.policy_net = DQN(state_dim, action_dim).to(self.device)  # Main network
        self.target_net = DQN(state_dim, action_dim).to(self.device)  # Target network (for stability)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Sync initially
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)  # Adam optimizer
        self.steps_done = 0  # Step counter for epsilon decay

    def select_action(self, state):
        # Epsilon-greedy strategy: exploration vs exploitation
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if random.random() > epsilon:
            with torch.no_grad():  # No gradient tracking for inference
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.policy_net(state_tensor).argmax(dim=1).item()  # Choose best action
        else:
            return random.randrange(2)  # Random action (exploration)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Don't train until enough samples are collected

        batch = self.memory.sample(BATCH_SIZE)  # Sample a batch
        batch = list(zip(*batch))  # Unzip batch

        # Convert batch data to tensors
        states = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch[1], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)  # Q-values of selected actions
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]  # Max Q-value for next state
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)  # Compute target Q-values

        loss = F.mse_loss(q_values, target_q_values)  # Mean squared error loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights to target net

# -------------------------------
# Training function
# -------------------------------
def train(num_episodes):
    env = gym.make("CartPole-v1", render_mode=None)  # Initialize gym environment
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    training_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):  # Limit steps per episode
            action = agent.select_action(state)
            next_state, reward, terminated ,truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.optimize_model()
            if done:
                break

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        training_rewards.append(total_reward)
        print(f"Episode {episode}: Reward {total_reward}")

    env.close()
    return agent, training_rewards

# -------------------------------
# Testing function
# -------------------------------
def test(agent, num_episodes, render):
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    evaluation_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            action = agent.policy_net(state_tensor).argmax().item()  # Always pick best action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 如果任一为 True，则 episode 结束
            state = next_state
            total_reward += reward
            if reward >= 500:
                done = True

        evaluation_rewards.append(total_reward)
        print(f"Test Episode {episode}: Reward {total_reward}")

    env.close()
    return evaluation_rewards

# -------------------------------
# Plot results
# -------------------------------
def plot_rewards_together(training_rewards, evaluation_rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(training_rewards, label='Training Rewards', color='blue')
    x_eval = list(range(len(training_rewards), len(training_rewards) + len(evaluation_rewards)))
    plt.plot(x_eval, evaluation_rewards, label='Evaluation Rewards', color='red')
    plt.title('Training & Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# Main program
# -------------------------------
if __name__ == "__main__":
    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 64
    MEMORY_SIZE = 10000
    TARGET_UPDATE = 20
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 500
    print("Starting DQN training...")
    trained_agent, training_rewards = train(num_episodes=2000)
    print("\n=== Testing Phase ===")
    evaluation_rewards = test(trained_agent, num_episodes=200, render=False)
    plot_rewards_together(training_rewards, evaluation_rewards)
