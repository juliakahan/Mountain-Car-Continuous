import gymnasium as gym
import numpy as np

# Ustawienia algorytmu Q-learning
alpha = 0.2  # Współczynnik uczenia
gamma = 0.99  # Współczynnik dyskontowania
epsilon = 0.1  # Parametr eksploracji
episodes = 5000  # Liczba epizodów

# Dyskretyzacja stanów
def discretize_state(state):
    state = np.around((state - env.observation_space.low) * np.array([10, 10, 5, 5]) / (env.observation_space.high - env.observation_space.low))
    return tuple(state.astype(int))

# Inicjalizacja tablicy Q
def init_q_table():
    q_table = {}
    for i in range(-100, 101):
        for j in range(-100, 101):
            for k in range(-50, 51):
                for l in range(-50, 51):
                    q_table[(i, j, k, l)] = [0, 0]
    return q_table

# Wybór akcji zgodnie z strategią e-greedy
def get_action(state, Q, epsilon):
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    return action

# Uruchomienie środowiska
env = gym.make('MountainCarContinuous-v0')

# Inicjalizacja tablicy Q
Q = init_q_table()

# Pętla ucząca
for episode in range(episodes):
    state = env.reset()
    state = discretize_state(state)
    done = False
    while not done:
        action = get_action(state, Q, epsilon)
        next_state, reward, done, _ = env.step([action])
        next_state = discretize_state(next_state)
        max_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * Q[next_state][max_next_action] - Q[state][action])
        state = next_state

    # Wypisz wynik co 100 epizodów
    if episode % 100 == 0:
        print("Episode:", episode)

# Testowanie agenta
state = env.reset()
state = discretize_state(state)
done = False
total_reward = 0
while not done:
    action = np.argmax(Q[state])
    next_state, reward, done, _ = env.step([action])
    next_state = discretize_state(next_state)
    state = next_state
    total_reward += reward

print("Total reward:", total_reward)
