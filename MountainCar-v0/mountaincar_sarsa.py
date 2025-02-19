import numpy as np
import random
import time
import matplotlib.pyplot as plt
import gym


# Utility Functions
def create_bins():
    """Create discretization bins for the state space."""
    n_bins = 20  # Adjust for finer/coarser discretization
    bins = [
        np.linspace(-1.2, 0.6, n_bins - 1),   # bins for position
        np.linspace(-0.07, 0.07, n_bins - 1)    # bins for velocity
    ]
    return bins


def discretize_state(state, bins):
    """Convert a continuous state into discrete indices."""
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))


# SARSA Algorithm Implementation
def sarsa(env, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
    """Train the agent using the SARSA algorithm."""
    bins = create_bins()
    n_bins = [len(b) + 1 for b in bins]
    n_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q_table = np.zeros(n_bins + [n_actions])
    rewards_per_episode = []
    max_steps = 200  # limit the maximum steps per episode

    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = discretize_state(state, bins)
        total_reward = 0
        done = False
        step_count = 0  # step counter

        # Choose the initial action using epsilon-greedy policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[discrete_state])

        # Run the episode for at most max_steps steps
        while not done and step_count < max_steps:
            step_count += 1

            # Take action and observe result
            next_state, reward, done, _, _ = env.step(action)
            discrete_next_state = discretize_state(next_state, bins)

            # Choose the next action using epsilon-greedy policy
            if random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q_table[discrete_next_state])

            # SARSA update
            td_target = reward + gamma * Q_table[discrete_next_state][next_action]
            Q_table[discrete_state][action] += alpha * (td_target - Q_table[discrete_state][action])

            discrete_state = discrete_next_state
            action = next_action
            total_reward += reward

        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}  Total Reward: {total_reward:.2f}  Current Epsilon: {epsilon:.3f}")

    return Q_table, rewards_per_episode


# Policy Evaluation Function
def evaluate_policy(env, Q_table, bins, num_episodes, render):
    """Evaluate the learned policy over several episodes."""
    total_rewards = 0
    test_rewards_per_episode = []
    max_steps = 200  # limit the maximum steps per episode

    if render:
        env = gym.make("MountainCar-v0", render_mode="human")
    else:
        env = gym.make("MountainCar-v0")

    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = discretize_state(state, bins)
        done = False
        episode_reward = 0
        step_count = 0  # step counter

        # Run the episode for at most max_steps steps
        while not done and step_count < max_steps:
            step_count += 1
            # Always choose the best action (no exploration)
            action = np.argmax(Q_table[discrete_state])
            state, reward, done, _, _ = env.step(action)
            if render:
                env.render()
                time.sleep(0.02)
            discrete_state = discretize_state(state, bins)
            episode_reward += reward

        test_rewards_per_episode.append(episode_reward)
        total_rewards += episode_reward
        print(f"Evaluation Episode {episode + 1}: Reward {episode_reward}")

    avg_reward = total_rewards / num_episodes
    print(f"Evaluation Average Reward: {avg_reward:.2f}")
    return test_rewards_per_episode


def plot_rewards_together(training_rewards, evaluation_rewards):
    """Plot training and evaluation rewards on the same graph."""
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


# Main function
if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # Hyperparameters (tuning might be needed for MountainCar)
    num_episodes = 6000    # many episodes might be needed for MountainCar
    alpha = 0.1            # learning rate
    gamma = 0.99           # discount factor
    epsilon = 1.0          # initial exploration rate
    epsilon_min = 0.01     # minimum exploration rate
    epsilon_decay = 0.995  # epsilon decay rate
    test_num_episodes = 200  # number of evaluation episodes

    print("Starting SARSA training on MountainCar-v0...")
    Q_table, training_rewards = sarsa(env, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    print("Training complete!")

    bins = create_bins()
    print("\nEvaluating the trained policy...")
    env = gym.make("MountainCar-v0")
    evaluation_rewards = evaluate_policy(env, Q_table, bins, test_num_episodes, render=False)
    print("Evaluation rewards:", evaluation_rewards)
    env.close()

    # Plot the training and evaluation rewards
    plot_rewards_together(training_rewards, evaluation_rewards)
