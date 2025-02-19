import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import gym

# def set_seed(env, seed):
#     """
#     Set random seeds for reproducibility.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     env.reset(seed=seed)
#     env.action_space.seed(seed)


def create_bins():
    """
    Create bins to discretize the continuous state space.
    """
    n_bins = 6
    bins = [
        np.linspace(-2.4, 2.4, n_bins - 1),  # cart position
        np.linspace(-3.0, 3.0, n_bins - 1),  # cart velocity
        np.linspace(-0.418, 0.418, n_bins - 1),  # pole angle
        np.linspace(-3.5, 3.5, n_bins - 1)  # pole angular velocity
    ]
    return bins


def discretize_state(state, bins):
    """
    Convert a continuous state into a discrete state by using the bins.
    """
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))


# ---------------------------
# 2. SARSA Algorithm
# ---------------------------
def sarsa(env, num_episodes, alpha, gamma,
          epsilon, epsilon_min, epsilon_decay):
    """
    Train the agent using the SARSA algorithm.

    :param env: The environment (e.g., CartPole)
    :param num_episodes: Number of training episodes
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon: Initial exploration rate
    :param epsilon_min: Minimum exploration rate
    :param epsilon_decay: Decay rate for epsilon
    :return: Q-table and list of rewards per episode
    """
    bins = create_bins()
    # Number of discrete states for each dimension
    n_bins = [len(b) + 1 for b in bins]
    n_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q_table = np.zeros(n_bins + [n_actions])
    rewards_per_episode = []

    for episode in range(num_episodes):
        # Reset environment and discretize initial state
        state, _ = env.reset()
        discrete_state = discretize_state(state, bins)

        # Choose an action using epsilon-greedy from the start state
        action = (env.action_space.sample()
                  if random.uniform(0, 1) < epsilon
                  else np.argmax(Q_table[discrete_state]))

        total_reward = 0
        done = False

        while not done:
            # Take the chosen action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Discretize the next state
            discrete_next_state = discretize_state(next_state, bins)

            # Choose next action using epsilon-greedy policy (SARSA uses the next action actually chosen)
            next_action = (env.action_space.sample()
                           if random.uniform(0, 1) < epsilon
                           else np.argmax(Q_table[discrete_next_state]))

            # Compute TD target for SARSA
            td_target = reward + gamma * Q_table[discrete_next_state][next_action]

            # Update Q-value
            Q_table[discrete_state][action] += alpha * (td_target - Q_table[discrete_state][action])

            # Transition to next state
            discrete_state = discrete_next_state
            action = next_action
            total_reward += reward

            # Optionally end early if some criterion is met
            if total_reward == 500:
                done = True

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        rewards_per_episode.append(total_reward)

        # Print update every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward} | "
                  f"Epsilon: {epsilon:.3f}")

    return Q_table, rewards_per_episode


# ---------------------------
# 3. Policy Evaluation
# ---------------------------
def evaluate_policy(env, Q_table, bins, num_episodes, render):
    """
    Evaluate the learned policy over multiple episodes.

    :param env: The environment
    :param Q_table: The learned Q-table
    :param bins: Discretization bins
    :param num_episodes: Number of evaluation episodes
    :param render: Whether to render the environment visually
    """
    total_rewards = 0
    test_rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = discretize_state(state, bins)
        done = False
        episode_reward = 0

        while not done:
            # Choose best action (greedy) during evaluation
            action = np.argmax(Q_table[discrete_state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if render:
                env.render()
                time.sleep(0.02)

            discrete_state = discretize_state(state, bins)
            episode_reward += reward

            # If you want to stop early when 500 is reached:
            if episode_reward == 500:
                done = True

        test_rewards_per_episode.append(episode_reward)
        total_rewards += episode_reward
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")

    avg_reward = total_rewards / num_episodes
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    return test_rewards_per_episode


def plot_rewards_together(training_rewards, evaluation_rewards):
    """
    Plot training and test rewards on the same figure.
    """
    plt.figure(figsize=(8, 4))

    # Plot training rewards
    plt.plot(training_rewards, label='Training Rewards', color='blue')

    # Plot evaluation rewards (appended after training episodes)
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
# 4. Main
# ---------------------------
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    #set_seed(env, 42)

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # Hyperparameters
    num_episodes = 2000
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    test_num_episodes = 200

    # 1) Train with SARSA
    print("Starting SARSA training...")
    Q_table, training_rewards = sarsa(env,
                                      num_episodes,
                                      alpha,
                                      gamma,
                                      epsilon,
                                      epsilon_min,
                                      epsilon_decay)
    print("Training complete!")

    # 2) Evaluate the trained policy
    bins = create_bins()
    print("\nEvaluating trained policy...")


    Visualization_of_Test=0
    bins = create_bins()
    if(Visualization_of_Test==0):
        env = gym.make("CartPole-v1")
        eval_rewards = evaluate_policy(env, Q_table, bins, test_num_episodes, render=False)

    else:
        env = gym.make("CartPole-v1", render_mode="human")
        eval_rewards = evaluate_policy(env, Q_table, bins, test_num_episodes, render=True)

    env.close()

    # 3) Plot training vs evaluation rewards
    plot_rewards_together(training_rewards, eval_rewards)
