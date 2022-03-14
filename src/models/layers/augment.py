import tensorflow as tf

class Augment(tf.keras.layers.Layer):

  def __init__(self, seed: int = 42) -> None:
    super().__ini__()
    self.augment_inputs = tf.keras.layers.RandomFlip(mode='horizonal', seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode='horizonal', seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels
