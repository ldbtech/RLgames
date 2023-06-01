import numpy as np
from collections import deque
import tensorflow as tf
from .NN import Actor


class DeepQ:
    def __init__(
        self,
        state_size,
        action_size,
        epsilon_decay,
        learning_rate,
        discount_factor,
        epsilon_min,
        memory_size,
    ):
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_min = epsilon_min
        self.epsilon = 1.0
        self.memory = deque(maxlen=memory_size)
        self.state_size = state_size
        self.action_size = action_size
        self.model = Actor(self.state_size, self.action_size)

    def convert_action_to_coords(self, action):
        col = action % 3
        row = action // 3
        return (col, row)

    def generate_random_coords(self):
        coords = []
        col = np.random.randint(0, 3)
        row = np.random.randint(0, 3)
        coords.append((col, row))

        return coords

    def convert_action_to_coords(self, action):
        col = action % 3
        row = action // 3
        return (col, row)

    # Transition is in form of tuple containing: state, action, reward, next_state, done
    def remember(self, transition):
        self.memory.append((transition))

    def takeAction(self, state):
        if np.random.rand() <= self.epsilon:
            return self.generate_random_coords()

        action_value = self.model.predict(state)
        return self.convert_action_to_coords(np.argmax(action_value[0]))

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for transition in minibatch:
            target = transition[2]
            if not transition[4]:
                target = transition[2] + self.discount_factor * np.amax(
                    self.model.predict(transition[3])[0]
                )
            target_f = self.model.predict(transition[0])
            target_f[0][transition[1]] = target
            self.model.fit(transition[0], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, filename):
        self.model.load_weights(filename)

    def save_model(self, filename):
        self.model.save_weights(filename)
