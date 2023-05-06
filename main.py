import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#nn model

model = Sequential([
    Dense(64, input_shape=(2,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='tanh')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

#training
gym.make('MountainCarContinuous-v0')


