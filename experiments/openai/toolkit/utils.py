import random
import gym
import numpy as np
from typing import Set, List
import matplotlib.pyplot as plt

def make_episodes_stats_plot(num_episodes: int, penaltiesList: List, epochsList: List):
    plt.plot(list(range(1,num_episodes)), penaltiesList, color='green', label="Penalties")
    plt.plot(list(range(1,num_episodes)), epochsList, color='red', label="Epochs")
    plt.xlabel("Episodes")
    plt.legend(loc="best")

def train(env: gym.Env, validPenalties: Set, q_table: np.ndarray, num_episodes: int=5):
    terminated = False
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    penaltiesList = []
    epochsList = []

    for _ in range(1,num_episodes):
        state = env.reset()
        epochs, penalties = 1, 0
        while not terminated:
            if random.uniform(0, 1) < epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            # Q Table update formula
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward in validPenalties:
                penalties += 1

            state = next_state
            epochs += 1
        
        penaltiesList.append(penalties)
        epochsList.append(epochs)
        
    total_penalties = np.sum(penaltiesList)
    total_epochs = np.sum(epochsList)
    make_episodes_stats_plot(num_episodes, penaltiesList, epochsList)

    return total_epochs, total_penalties

def evaluate(env: gym.Env, validPenalties: Set, q_table: np.ndarray=None, num_episodes: int=5):
    penaltiesList = []
    epochsList = []

    for _ in range(1,num_episodes):
        state = env.reset()
        epochs, penalties = 1, 0
        terminated = False
        
        while not terminated:
            if (q_table is not None): 
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)

            if reward in validPenalties:
                penalties += 1

            epochs += 1

        penaltiesList.append(penalties)
        epochsList.append(epochs)
        
    total_penalties = np.sum(penaltiesList)
    total_epochs = np.sum(epochsList)
    make_episodes_stats_plot(num_episodes, penaltiesList, epochsList)
    
    return total_epochs, total_penalties