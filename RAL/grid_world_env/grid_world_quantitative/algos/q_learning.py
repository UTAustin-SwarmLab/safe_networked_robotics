import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import utils.plotting as plotting

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, shield = False):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    global Q
    Q = defaultdict(lambda: np.zeros(env.n_action))
    # Q = defaultdict(defaultdict(np.zeros(env.n_action)).copy)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        game_status = np.zeros(3, dtype=np.int32))    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.n_action)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, info = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                if info['gamestatus'] == 'won':
                    stats.game_status[0] += 1
                elif info['gamestatus'] == 'lost':
                    stats.game_status[1] += 1
                else:
                    stats.game_status[2] += 1
                break
                
            state = next_state
    
    return Q, stats

def q_learning_test_pm(env, Q, epsilon = 0.0, num_episodes = 100):
    policy = make_epsilon_greedy_policy(Q, epsilon, env.n_action)
    stats = plotting.TestStats_pm(robot=[],
                                  adversary=[],
                                  reward=[], game_status=np.zeros(3))

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        reward_to_go = 0
        while not done:
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            state, reward, done, info = env.step(action)
            
            if i_episode == num_episodes-1:
                reward_to_go += reward
                stats.robot.append(info["robot"])
                stats.adversary.append(info["adversary"])
                stats.reward.append(reward_to_go)

            if done:
                if info['gamestatus'] == 'won':
                    stats.game_status[0] += 1
                elif info['gamestatus'] == 'lost':
                    stats.game_status[1] += 1
                else:
                    stats.game_status[2] += 1
                break

    return stats