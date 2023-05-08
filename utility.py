import numpy as np
import matplotlib.pyplot as plt
import csv

# number of buckets for position and velocity
pos_bucket_numer = 20
vel_bucket_number = 20

# There are 3 discrete deterministic actions:
# 0: Accelerate to the left
# 1: Don’t accelerate
# 2: Accelerate to the right
action_space = [0, 1, 2]

# bins for pos and vel
# position is between [-1.2, 0.6]
# and velocity [-0.07, 0.07]
pos_space = np.linspace(-1.2, 0.6, pos_bucket_numer)
vel_space = np.linspace(-0.07, 0.07, vel_bucket_number)

# continuous observation -> discrete one
def get_state(observation):
    # Cart Position, Cart Velocity
    (pos, vel) = observation
    # return bins of current observation state
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))

    return (pos_bin, vel_bin)

# find max action from Q (Q is a dict)
def max_action(Q, state, actions=action_space):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)

    return action

# credit: J.K.
def plot(scores, epsilons, num_of_episodes):
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

def save_csv(scores, epsilons, num_of_episodes):
    data = zip(scores, epsilons)

    with open('scores_epsilons.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Score', 'Epsilon'])
        writer.writerows(data)
