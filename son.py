from controller import Supervisor, Keyboard
from vehicle import Driver
from math import pi, isinf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Input
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
import tensorflow as tf
import math
import random
import numpy as np

# Robot aynı zamanda Supervisor olduğu için Supervisor sınıfı kullanılır
robot = Supervisor()
driver = Driver()

timestep = int(robot.getBasicTimeStep())

# Klavye ve sensörler
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

gps = robot.getDevice('gps')
gps.enable(timestep)

accelerometer = robot.getDevice('accelerometer')
accelerometer.enable(timestep)

imu = robot.getDevice('inertial unit')
imu.enable(timestep)

lidar_front = robot.getDevice('lidar_front')
lidar_front.enable(timestep)

lidar_left = robot.getDevice('lidar_left')
lidar_left.enable(timestep)

lidar_right = robot.getDevice('lidar_right')
lidar_right.enable(timestep)

# Supervisor olarak kendi node'unu al
vehicle_node = robot.getSelf()
translation_field = vehicle_node.getField("translation")
rotation_field = vehicle_node.getField("rotation")

initial_position = translation_field.getSFVec3f()
initial_rotation = rotation_field.getSFRotation()

# Sürüş parametreleri
min_steering = -0.15 * pi
max_steering = 0.15 * pi
max_speed = 10
min_speed = -10
speed_step = 0.1
steering_step = 0.04

speed = 0
steering = 0

latest_lidar_data = [-1, -1, -1, -1, -1, -1]

#Input Shape ve Action Size
INPUT_SHAPE = (14,) #emin değilim
ACTION_SIZE = 4

#Eğitim sabitleri
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "DNN"
MINIBATCH_SIZE = 32
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -200
MEMORY_FRACTION = 0.20
SAVE_EVERY = 1000
EPISODES = 20_000
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#Başlangıç epsilon değeri
epsilon = 1

#Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

node = robot.getFromDef("TARGET")
goal_position = node.getField("translation").getSFVec3f()

# Reset fonksiyonu
def reset_vehicle():
    translation_field.setSFVec3f(initial_position)
    rotation_field.setSFRotation(initial_rotation)
    robot.simulationResetPhysics()
    print("Vehicle position has been reset.")
    lidar_data = [-1, -1, -1, -1, -1, -1]
    state = [*initial_position,
             *accelerometer.getValues(),
             *imu.getRollPitchYaw(),
             *lidar_data,
             *goal_position
             ]
    return state

#model fonksiyonları
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

def is_done(current_pos, step, goal_pos) -> bool:
    if step >= 200:
        return True
    if current_pos == goal_pos:
        return True
    return False

def reward_function(lidar_data, current_pos, goal_pos, action, imu_data) -> float:
    roll, pitch, yaw = imu_data
    reward = 0.0

    min_distance = min(lidar_data)
    if min_distance == 0:
        reward -= 100
    elif min_distance < 10:
        reward -= 10

    def euclidean(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    distance_to_goal = euclidean(current_pos, goal_pos)
    reward -= distance_to_goal * 0.1

    # Steering angle yerine yaw kullanılıyor
    if action == 1 and yaw > 30:
        reward += 5
    elif action == 2 and yaw < -30:
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

#models
dqn_model = create_model(INPUT_SHAPE, ACTION_SIZE)
target_model = create_model(INPUT_SHAPE, ACTION_SIZE)
target_model.set_weights(dqn_model.get_weights())
#replay buffer
replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

# Ana döngü
for episode in tqdm(range(EPISODES), desc="Training progress", unit="episodes"):
    t = 1
    done = False
    last_state = reset_vehicle()
    total_reward = 0

    #while robot.step(timestep) != -1 or not done: bunu değiştim
    while not done:

        t += 1

        gps_data = gps.getValues()
        accelerometer_data = accelerometer.getValues()
        imu_data = imu.getRollPitchYaw()

        # Lidar verisi seyrek alınabilir
        if t % 16 == 0:
            ranges_left = lidar_left.getRangeImage()
            ranges_right = lidar_right.getRangeImage()
            ranges_front = lidar_front.getRangeImage()
            latest_lidar_data = ranges_left + ranges_right + ranges_front
            for i in range(len(latest_lidar_data)):
                if (isinf(latest_lidar_data[i])):
                    latest_lidar_data[i] = -1
            t = 1
        t += 1
        # current state'e hedef konumu ekledim
        current_state = [
            *gps_data,
            *accelerometer_data,
            *imu_data,
            *latest_lidar_data
             * goal_position
        ]

        action = epsilon_greedy_policy(dqn_model, current_state, epsilon, ACTION_SIZE)
        reward = reward_function(current_state, gps_data, goal_position, action, imu_data)
        total_reward += reward

        # Store experience in replay buffer
        replay_buffer.add((current_state, action, reward, current_state, done))
        current_state = last_state

        # Sample random minibatch from buffer and update model
        if replay_buffer.size() >= MINIBATCH_SIZE:
            minibatch = replay_buffer.sample(MINIBATCH_SIZE)
            train_minibatch(dqn_model, target_model, minibatch, DISCOUNT)

        # Update target model periodically
        if episode % 10 == 0:
            target_model.set_weights(dqn_model.get_weights())

        # Decay epsilon
        epsilon = max(epsilon * EPSILON_DECAY, MIN_EPSILON)

        if not episode % SAVE_EVERY:
            dqn_model.save(f'models/dqn_model_episode_{episode}.keras')

        print(f"Episode: {episode}, Reward: {total_reward}")

    # Driver API ile hız ve direksiyon kontrolü
    driver.setCruisingSpeed(speed)
    driver.setSteeringAngle(steering)




