# OpenAI Vectorized Environments
import gym
import numpy as np

envs = gym.vector.make("LunarLander-v2", num_envs=3, render_mode = "human")
envs.reset()

for _ in range(1000):
    observation, reward, done, info = envs.step(envs.action_space.sample())

    if done.all():
        observation, info = envs.reset()

envs.close()