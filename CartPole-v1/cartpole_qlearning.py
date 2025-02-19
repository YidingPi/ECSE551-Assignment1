import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
# Import the local CartPole implementation
#from cartpole import CartPoleEnv  # Ensure cartpole.py contains the provided implementation
import gym ## Import OpenAI Gym for the CartPole environment

# Set a fixed seed for reproducibility
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     env.reset(seed=seed)
#     env.action_space.seed(seed)

# discretizing the state space
def create_bins():
    n_bins = 6
    # n_bin help to discretizing continuous values into discrete states.
    bins = [
        np.linspace(-2.4, 2.4, n_bins - 1),   # cart position
        np.linspace(-3.0, 3.0, n_bins - 1),   # cart speed
        np.linspace(-0.418, 0.418, n_bins - 1),  # pole angular
        np.linspace(-3.5, 3.5, n_bins - 1)    # pole angular velocity
    ]
    return bins

def discretize_state(state, bins):
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))
# Q-Learning Algorithm Implementation
def q_learning(env, num_episodes, alpha, gamma,
               epsilon, epsilon_min, epsilon_decay):
    """
    Train the agent using Q-learning.
    :param env: The environment
    :param num_episodes: Number of training episodes
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon: Initial exploration rate
    :param epsilon_min: Minimum exploration rate
    :param epsilon_decay: Decay rate for epsilon (exploration-exploitation tradeoff)
    """
    bins = create_bins()
    n_bins = [len(b) + 1 for b in bins]
    n_actions = env.action_space.n
    # Initialize the Q-table with zeros
    Q_table = np.zeros(n_bins + [n_actions])
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = discretize_state(state, bins)
        total_reward = 0
        done = False

        while not done:
            # Select action using epsilon-greedy policy
            action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(Q_table[discrete_state])
            # Execute action and observe the next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            discrete_next_state = discretize_state(next_state, bins)
            # Q-learning update rule
            best_next_action = np.argmax(Q_table[discrete_next_state])
            td_target = reward + gamma * Q_table[discrete_next_state][best_next_action]
            Q_table[discrete_state][action] += alpha * (td_target - Q_table[discrete_state][action])

            discrete_state = discrete_next_state
            total_reward += reward
            # Stop episode early if max reward is reached
            if(total_reward==500):
                  done=True
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}，Reward: {total_reward}，current epsilon: {epsilon:.3f}")

    return Q_table, rewards_per_episode

# ---------------------------
# Policy Evaluation Function
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
            action = np.argmax(Q_table[discrete_state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # if render:
                #env.render()
                #time.sleep(0.02)  #slow the speed of action
            discrete_state = discretize_state(state, bins)
            episode_reward += reward
            if(episode_reward)==500:
                 done=True
        test_rewards_per_episode.append(episode_reward)
        total_rewards += episode_reward
        print(f"Evaluation Episode {episode+1}：Reward {episode_reward}")

    avg_reward = total_rewards / num_episodes
    print(f"Evaluation Average Reward：{avg_reward}")
    return test_rewards_per_episode
def plot_rewards_together(training_rewards, evaluation_rewards):
    """
    Plot training and test result
    """
    plt.figure(figsize=(8, 4))

    plt.plot(training_rewards, label='Training Rewards', color='blue')

    start_index = len(training_rewards)
    x_eval = list(range(start_index, start_index + len(evaluation_rewards)))
    plt.plot(x_eval, evaluation_rewards, label='Evaluation Rewards', color='red')

    plt.title('Training & Evaluation Rewards in One Figure')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.show()
# ---------------------------
# main function
# ---------------------------
if __name__ == '__main__':
    #env = CartPoleEnv()  # if use local cartpole
    #env = gym.make("CartPole-v1")
    #env = gym.make("CartPole-v1",render_mode="human")
    #set_seed(42)
    num_episodes = 2000
    alpha = 0.10
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    test_num_episodes = 200
    print("Starting Q-Learning training...")
    Visualization_of_Learning = 0
    if(Visualization_of_Learning==0):
        env = gym.make("CartPole-v1")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
        Q_table, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    # else:
    #     env = gym.make("CartPole-v1", render_mode="human")
    #     print(f"Observation Space: {env.observation_space}")
    #     print(f"Action Space: {env.action_space}")
    #     Q_table, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    print("Training complete!")
    # env = CartPoleEnv("human") // if import local game program
    Visualization_of_Test=0
    bins = create_bins()
    if(Visualization_of_Test==0):
        env = gym.make("CartPole-v1")
        eval_rewards = evaluate_policy(env, Q_table, bins, test_num_episodes, render=False)
    else:
        env = gym.make("CartPole-v1", render_mode="human")
        eval_rewards = evaluate_policy(env, Q_table, bins, test_num_episodes, render=False)
    print("\nEvaluating trained policy...")
    env.close()

    plot_rewards_together(rewards, eval_rewards) #plot learning curves