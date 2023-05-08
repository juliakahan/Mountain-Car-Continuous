import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

from utility import *

if __name__ == '__main__':
    num_of_episodes = 5000
    num_of_steps_per_episode = 1000
    alpha = 0.01  # discount parameter
    gamma = 0.9  # discount parameter
    eps = 1.  # exploration parameter

    # env = gym.make('MountainCar-v0', render_mode="human")
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

        mean_score = np.mean(scores[max(0, episode - 100):(episode + 1)])
        if episode % 100 == 0:
            print("Episode:", episode, "Score:", score, "epsilon:", eps)
            print("Mean score over last 100 episodes:", mean_score)


    env.close()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 12), sharex=True)

    ax1.plot(scores, color='blue')
    ax1.set_xlabel('Episode', fontsize=16)
    ax1.set_ylabel('Score', color='blue', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=14)
    ax1.set_ylim(-1000, 0)
    ax1.set_yticks(range(-1000, 1, 100))
    ax1.set_xlim(0, num_of_episodes)
    ax1.set_xticks(np.arange(0, num_of_episodes, 100))
    ax1.grid(alpha=0.4)

    ax2.plot(epsilons, color='red', linewidth=3)
    ax2.set_ylabel('Epsilon', color='red', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=14)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.4)

    mean_scores = [np.mean(scores[max(0, i - 99):(i + 1)]) for i in range(len(scores))]
    ax3 = ax1.twinx()
    ax3.plot(mean_scores, color='green', linewidth=3)
    ax3.set_ylabel('Mean score (last 100 episodes)', color='green', fontsize=16)
    ax3.tick_params(axis='y', labelcolor='green', labelsize=14)
    ax3.set_ylim(-1000, 0)
    ax3.set_yticks(range(-1000, 1, 100))
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()

    plt.title('num_of_episodes = ' + str(num_of_episodes), fontsize=20, color='green')

    plt.tight_layout()
    plt.savefig('scores_mcc.png')
    plt.show()



    f = open("mountaincar.pkl", "wb")
    pickle.dump(Q, f)
    f.close()
