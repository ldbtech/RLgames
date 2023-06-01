import tensorflow as tf


class Actor(tf.keras.Model):
    """
    This is a neural network that will be used in different models.
    Please be advised that this is a simple forward prop NN; we can make it deeper
    after finishing the initial model.
    """

    def __init__(self, input_size, num_actions):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_actions, activation="linear"),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")

    def call(self, inputs):
        return self.model(inputs)
