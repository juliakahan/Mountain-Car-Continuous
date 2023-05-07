import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

from assets import *

if __name__ == '__main__':
    num_of_steps_per_episode = 1000

    env = gym.make('MountainCar-v0', render_mode="human")

    pickle_in = open('mountaincar.pkl', 'rb')
    Q = pickle.load(pickle_in)

    terminated = False
    truncated = False

    obs, _ = env.reset()
    state = get_state(obs)

    score = 0

    while not (terminated or truncated):
        action = max_action(Q, state)

        obs_new, reward, terminated, truncated, info = env.step(action)
        state = get_state(obs_new)
        score += reward

    print("Score:", score)

    env.close()

