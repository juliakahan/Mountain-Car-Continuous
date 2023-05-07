import numpy as np

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
