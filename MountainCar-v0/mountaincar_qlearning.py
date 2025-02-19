import numpy as np
import random
import time
import matplotlib.pyplot as plt
import gym


# ---------------------------
# Utility Functions
# ---------------------------
def create_bins():
    # Create discretization bins for position and velocity
    n_bins = 20  # Adjust the granularity as needed
    bins = [
        np.linspace(-1.2, 0.6, n_bins - 1),  # Position bins
        np.linspace(-0.07, 0.07, n_bins - 1)  # Velocity bins
    ]
    return bins


def discretize_state(state, bins):
    # Convert continuous state variables into discrete bins
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))


# ---------------------------
# Q-Learning Algorithm Implementation
# ---------------------------
def q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
    """
    Train an agent using the Q-learning algorithm.

    :param env: MountainCar environment
    :param num_episodes: Number of training episodes
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon: Initial exploration rate
    :param epsilon_min: Minimum exploration rate
    :param epsilon_decay: Decay rate for epsilon
    """
    bins = create_bins()
    # Number of discrete bins per dimension is len(bins) + 1
    n_bins = [len(b) + 1 for b in bins]
    n_actions = env.action_space.n  # For MountainCar, usually 3 actions
    # Initialize Q-table with shape [position_bins, velocity_bins, actions]
    Q_table = np.zeros(n_bins + [n_actions])
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = discretize_state(state, bins)
        total_reward = 0
        done = False
        step_count = 0  # Count steps in the episode

        while not done and step_count < 200:  # Limit each episode to 200 steps
            step_count += 1

            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[discrete_state])

            # Take the action
            next_state, reward, done, _, _ = env.step(action)
            discrete_next_state = discretize_state(next_state, bins)

            # Q-learning update
            best_next_action = np.argmax(Q_table[discrete_next_state])
            td_target = reward + gamma * Q_table[discrete_next_state][best_next_action]
            Q_table[discrete_state][action] += alpha * (td_target - Q_table[discrete_state][action])

            discrete_state = discrete_next_state
            total_reward += reward

        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}  Total Reward: {total_reward:.2f}  Current Epsilon: {epsilon:.3f}")

    return Q_table, rewards_per_episode


# ---------------------------
# Policy Evaluation Function
# ---------------------------
def evaluate_policy(env, Q_table, bins, num_episodes, render):
    """
    Evaluate the trained policy over several episodes.

    :param env: MountainCar environment
    :param Q_table: Trained Q-table
    :param bins: Discretization bins
    :param num_episodes: Number of evaluation episodes
    :param render: Whether to render the environment
    """
    total_rewards = 0
    test_rewards_per_episode = []
    if render:
        env = gym.make("MountainCar-v0", render_mode="human")
    else:
        env = gym.make("MountainCar-v0")
    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = discretize_state(state, bins)
        done = False
        episode_reward = 0
        step_count = 0  # Count steps in the episode

        while not done and step_count < 200:  # Limit each episode to 200 steps
            step_count += 1

            # Always select the best action (no exploration)
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
    """
    Plot training and evaluation rewards together.
    """
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
# Main function
# ---------------------------
if __name__ == '__main__':
    # Create the MountainCar environment
    env = gym.make("MountainCar-v0")
    # set_seed(42, env)

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # Hyperparameters (MountainCar usually needs many episodes to learn a good policy)
    num_episodes = 6000  # Number of training episodes
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.995  # Epsilon decay rate
    test_num_episodes = 200  # Number of evaluation episodes

    print("Starting Q-Learning training on MountainCar-v0...")
    Q_table, training_rewards = q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    print("Training complete!")

    bins = create_bins()
    print("\nEvaluating the trained policy...")
    # Set render=True if you want to see the evaluation visually
    evaluation_rewards = evaluate_policy(env, Q_table, bins, test_num_episodes, render=False)
    print("Evaluation rewards:", evaluation_rewards)
    env.close()

    plot_rewards_together(training_rewards, evaluation_rewards)
