from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Input
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
import tensorflow as tf

from BlobEnv import BlobEnv

import os
import numpy as np
import time
import random

from Blob import Blob
import numpy as np
from PIL import Image
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = False
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    OBSERVATION_SPACE_SIZE = 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}
    def __init__(self):
        pass
    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img

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


def create_model(self, input_shape, action_size=9):
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


env = BlobEnv()
INPUT_SHAPE = (env.OBSERVATION_SPACE_SIZE,)

dqn_model = create_model(env, INPUT_SHAPE)
target_model = create_model(env, INPUT_SHAPE)
target_model.set_weights(dqn_model.get_weights())
replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)


train_dqn(env, dqn_model, target_model, replay_buffer)