import matplotlib
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Input
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
import tensorflow as tf
import math

import os
import numpy as np
import time
import random

import numpy as np
from PIL import Image
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "DNN"
#MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200
MEMORY_FRACTION = 0.20
SAVE_EVERY = 1000

EPISODES = 20_000

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

SHOW_PREVIEW = False #try this with true


# For stats
ep_rewards = [-200]

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


def create_model(input_shape, action_size=9):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(action_size, activation='linear')  # Output Q-values
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def epsilon_greedy_policy(model, state, epsilon, action_size):
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)  # Random action (exploration)
    q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    return np.argmax(q_values)  # Action with max Q-value (exploitation)

def train_minibatch(dqn_model, target_model, minibatch, gamma):
    states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
    q_values = dqn_model.predict(states, verbose=0)
    q_next = target_model.predict(next_states, verbose=0)

    for i in range(len(minibatch)):
        q_values[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + gamma * np.max(q_next[i])

    dqn_model.fit(states, q_values, verbose=0)

def train_dqn(env, dqn_model, target_model, replay_buffer, episodes=EPISODES, gamma=DISCOUNT, batch_size=32, epsilon_start=1.0,
              epsilon_min=MIN_EPSILON, epsilon_decay=EPSILON_DECAY):
    epsilon = epsilon_start
    for episode in tqdm(range(episodes), desc="Training progress", unit="episodes"):
        state = env.reset()  # Reset environment at start of episode
        done = False
        total_reward = 0

        while not done:
            # Choose action
            action = epsilon_greedy_policy(dqn_model, state, epsilon, env.ACTION_SPACE_SIZE)

            # Take action, observe next_state, reward, and done flag
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store experience in replay buffer
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

            # Sample random minibatch from buffer and update model
            if replay_buffer.size() >= batch_size:
                minibatch = replay_buffer.sample(batch_size)
                train_minibatch(dqn_model, target_model, minibatch, gamma)

            # Update target model periodically
            if episode % 10 == 0:
                target_model.set_weights(dqn_model.get_weights())

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if not episode % SAVE_EVERY:
            dqn_model.save(f'models/dqn_model_episode_{episode}.keras')
        print(f"Episode: {episode}, Reward: {total_reward}")

def is_done(current_state: tuple, step: int, goal_state: tuple) -> bool:
    if step >= 200:
        return True
    if current_state == goal_state:
        return True
    return False

def reward_function(
    sensors: list,
    orientation: int,
    current_pos: tuple,
    goal_pos: tuple,
    steering_angle: int,
    action: int
) -> float:
    reward = 0.0

    min_distance = min(sensors)
    if min_distance == 0:
        reward -= 100
    elif min_distance < 10:
        reward -= 10

    def euclidean(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    distance_to_goal = euclidean(current_pos, goal_pos)
    reward -= distance_to_goal * 0.1

    if action == 1 and steering_angle > 30:
        reward += 5
    elif action == 2 and steering_angle < -30:
        reward += 5
    elif action in [1, 2]:
        reward -= 5

    if action == 3:
        reward += 1
    elif action == 4:
        reward += 1

    if distance_to_goal < 2.0:
        reward += 100

    return reward


#env = BlobEnv()
INPUT_SHAPE = (env.OBSERVATION_SPACE_SIZE,)

dqn_model = create_model(INPUT_SHAPE)
target_model = create_model(INPUT_SHAPE)
target_model.set_weights(dqn_model.get_weights())
replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)


train_dqn(env, dqn_model, target_model, replay_buffer)