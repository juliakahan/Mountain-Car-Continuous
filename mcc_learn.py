import gymnasium as gym
import numpy as np
from tqdm import tqdm
import pickle

from utility import *

if __name__ == '__main__':
    num_of_episodes = 5000
    num_of_steps_per_episode = 1000
    alpha = 0.1  # discount parameter
    gamma = 0.9  # discount parameter
    eps = 1.  # exploration parameter

    env = gym.make('MountainCar-v0', max_episode_steps=num_of_steps_per_episode)

    states = list()
    Q = {}

    scores = np.zeros(num_of_episodes)
    epsilons = np.zeros(num_of_episodes)

    for position in range(pos_bucket_numer + 1):
        for velocity in range(vel_bucket_number + 1):
            states.append((position, velocity))

    for state in states:
        for action in action_space:
            Q[state, action] = 0

    # create progress bar
    progress_bar = tqdm(total=num_of_episodes, desc='Learning')

    # learning loop
    for episode in range(num_of_episodes):
        terminated = False
        truncated = False

        obs, _ = env.reset()
        state = get_state(obs)

        score = 0

        while not (terminated or truncated):
            if np.random.random() < eps:
                action = np.random.choice(action_space)
            else:
                action = max_action(Q, state)

            obs_new, reward, terminated, truncated, info = env.step(action)
            score += reward

            # calculate Q
            state_new = get_state(obs_new)
            action_new = max_action(Q, state)
            Q[state, action] = Q[state, action] + alpha*(reward + gamma*Q[state_new, action_new] - Q[state, action])

            state = state_new

        # decrease epsilon over time (in halfway selection strategy will be almost entirely greedy)
        eps = eps - 2/num_of_episodes if eps > 0.01 else 0.01

        scores[episode] = score
        epsilons[episode] = eps

        # per one hundred episodes calculate mean score,
        # which is shown on the progress bar
        if episode % 100 == 0:
            # mean of last 100 episodes
            mean = np.mean(scores[max(0, episode - 100):(episode + 1)])

        # Update the progress bar
        progress_bar.set_postfix(epsilon=f'{eps:.2f}',
                                 score=str(int(score)),
                                 mean_score=str(int(mean)))
        progress_bar.update(1)

    env.close()

    plot(scores, epsilons, num_of_episodes)

    save_csv(scores, epsilons, num_of_episodes)

    f = open("mountaincar.pkl", "wb")
    pickle.dump(Q, f)
    f.close()
