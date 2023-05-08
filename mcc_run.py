import gymnasium as gym
import numpy as np
import pickle

from utility import *

# change to true if you can use Q from learning state
use_learned_Q = True

if __name__ == '__main__':
    num_of_steps_per_episode = 1000

    env = gym.make('MountainCar-v0',
                   max_episode_steps=num_of_steps_per_episode, render_mode="human")

    if use_learned_Q:
        pickle_in = open('mountaincar_new.pkl', 'rb')
        Q = pickle.load(pickle_in)

    terminated = False
    truncated = False

    obs, _ = env.reset()
    state = get_state(obs)

    score = 0

    while not (terminated or truncated):
        if use_learned_Q:
            action = max_action(Q, state)
        else:
            action = np.random.choice(action_space)

        obs_new, reward, terminated, truncated, info = env.step(action)
        state = get_state(obs_new)
        score += reward

    print("Score:", score)

    env.close()

