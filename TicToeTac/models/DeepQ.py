import numpy as np
from collections import deque


class DeepQ:
    def __init__(self, epsilon_decay, learning_rate, env, memory_size):
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.env = env
        self.epsilon = 1.0
        self.memory = deque(memory_size)

    def generate_random_coords(self):
        coords = []
        for _ in range(2):
            col = np.random.randint(0, 3)
            row = np.random.randint(0, 3)
            coords.append((col, row))

        while coords[0] == coords[1]:
            coords[1] = (np.random.randint(0, 3), np.random.randint(0, 3))

        return coords

    def takeAction(self, state):
        pass

    def train(self, episodes=350):
        for ep in range(episodes):
            total_rewards = 0
            done = False
            current_state = self.env.reset()
            while not done:
                action = self.takeAction(state=current_state)
                next_state, done, reward, info = self.env.step(action)
                total_rewards += reward

    def eval(self, episodes=10):
        for ep in episodes:
            pass
