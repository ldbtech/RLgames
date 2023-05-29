import tensorflow as tf


class Actor:
    def __init__(self, input_shape, num_actions):
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_actions, activation="linear"),
            ]
        )

    def forward(self):
        return self.model(self.input_shape)
