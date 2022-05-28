import unittest
import numpy as np
from tensorflow.python.eager.backprop import GradientTape
from src.models.losses.dice_loss import DiceError
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class TestDiceError(unittest.TestCase):

    def test_dice_error_with_multiple_class(self) -> None:

        np.random.seed(42)

        y_true = np.eye(3)
        y_true = np.append(y_true,np.eye(3))
        y_true = np.append(y_true,np.eye(3))
        y_true = y_true.reshape((1,3,3,3))
        y_pred = y_true

        dc = DiceError()
        loss = dc(y_true,y_pred)

        self.assertTrue(loss < 1e-5)

    def test_dice_error_min(self) -> None:

        np.random.seed(42)

        y_true = np.eye(3)
        y_true = y_true.reshape((1,3,3,1))
        y_pred = y_true

        dc = DiceError()
        loss = dc(y_true,y_pred)

        self.assertTrue(loss < 1e-5)

    def test_dice_error_max(self) -> None:

        np.random.seed(42)

        y_true = np.eye(3).reshape((1,3,3,1))
        y_pred = np.zeros((1,3,3,1))

        dc = DiceError(1)
        loss = dc(y_true,y_pred)

        self.assertTrue(loss > 0.75)

    def test_dice_error_med(self) -> None:

        np.random.seed(42)

        y_true = np.eye(3).reshape((1,3,3,1))
        y_pred = np.ones((1,3,3,1))

        dc = DiceError(1)
        loss = dc(y_true,y_pred)

        self.assertTrue(loss > 0.46 and loss < 0.47)

    def test_in_model(self) -> None:
        # Create model
        inputs = keras.Input(shape=(784,), name="digits")
        x1 = layers.Dense(64, activation="relu")(inputs)
        x2 = layers.Dense(64, activation="relu")(x1)
        outputs = layers.Dense(10, name="predictions")(x2)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Instantiate an optimizer.
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        # Instantiate a loss function
        loss_fn = DiceError()

        # Prepare the training dataset.
        batch_size = 64
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1,784))

        y_possibles = np.eye(y_train.max() + 1)
        y_train_binary = np.array([])
        for y in y_train:
            y_train_binary = np.append(y_train_binary, y_possibles[y])
        shape = (len(y_train), y_train.max() + 1)
        y_train_binary = np.reshape(y_train_binary, shape)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_binary))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size)

        epochs = 2
        for epoch in range(epochs):
            print(f"\n Start of epoch {epoch}")

            # Itearate over te batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, witch enables auto-differentiation.
                with GradientTape(True) as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    y_batch_pred = model(x_batch_train, training=True)

                    # Compute the loss value for this minibatch
                    loss_value = loss_fn(y_batch_train, y_batch_pred)

                # Use the gradient tape to aumotically retrieve
                # the gradients of th trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_variables)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Log every 200 batches.
                if step % 200 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("Seen so far: %s samples" % ((step + 1) * 64))


if __name__ == '__main__':
    unittest.main()
