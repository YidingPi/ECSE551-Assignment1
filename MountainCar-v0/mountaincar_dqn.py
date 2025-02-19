import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import time

# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)


# ---------------------------
# Q-Network Model
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Epsilon-Greedy Action Selection
# ---------------------------
def select_action(state, policy_net, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_t)
        _, action = torch.max(q_values, dim=1)
        return action.item()


# ---------------------------
# DQN Training Function
# ---------------------------
def train_dqn(env, episodes, gamma, lr, batch_size, buffer_capacity, min_buffer_size, epsilon_start, epsilon_end):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    training_rewards = []

    # Exponentially decay epsilon
    def get_epsilon(episode):
        return max(epsilon_end, epsilon_start * (0.995 ** episode))

    for episode in range(episodes):
        state, _ = env.reset()  # Gym >=0.26 returns (obs, info)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            epsilon = get_epsilon(episode)
            action = select_action(state, policy_net, epsilon, action_dim, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

            if len(replay_buffer) > min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update the target network every 100 steps
            if steps % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        training_rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}, Epsilon: {get_epsilon(episode):.2f}")

    return policy_net, training_rewards


# ---------------------------
# DQN Policy Evaluation Function
# ---------------------------
def evaluate(env, policy_net, episodes, render=False):
    if not render:
        env = gym.make("MountainCar-v0")
    else:
        env = gym.make("MountainCar-v0", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if render:
                env.render()
                time.sleep(0.02)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_t)
                _, action = torch.max(q_values, dim=1)
            action = action.item()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Evaluation Episode {ep + 1}, Reward: {episode_reward:.2f}")

    env.close()
    avg_reward = np.mean(rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    return rewards


# ---------------------------
# Plotting Function for Training and Evaluation Rewards
# ---------------------------
def plot_rewards_together(training_rewards, evaluation_rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(training_rewards, label='Training Rewards', color='blue')
    start_index = len(training_rewards)
    x_eval = list(range(start_index, start_index + len(evaluation_rewards)))
    plt.plot(x_eval, evaluation_rewards, label='Evaluation Rewards', color='red')
    plt.title('Training & Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main Function
# ---------------------------
if __name__ == "__main__":
    # Hyperparameters
    train_episodes = 2000
    eval_episodes = 200
    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    buffer_capacity = 100000
    min_buffer_size = 1000
    epsilon_start = 1.0
    epsilon_end = 0.01

    # Create the environment
    env = gym.make("MountainCar-v0")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    print("Starting DQN training on MountainCar-v0...")
    policy_net, training_rewards = train_dqn(env, train_episodes, gamma, lr, batch_size,
                                               buffer_capacity, min_buffer_size,
                                               epsilon_start, epsilon_end)
    print("Training complete!")

    # Create evaluation environment (set render=True to visualize)
    eval_env = gym.make("MountainCar-v0")
    print("\nEvaluating the trained policy...")
    evaluation_rewards = evaluate(eval_env, policy_net, eval_episodes, render=False)
    print("Evaluation complete!")

    # Plot training and evaluation rewards
    plot_rewards_together(training_rewards, evaluation_rewards)
