import unittest
import numpy as np
from tensorflow.python.keras.losses import BinaryCrossentropy
from src.models.losses.log_cosh_dice_loss import LogCoshDiceError

class TestLogCoshDiceError(unittest.TestCase):

    def test_lchdice_error_min(self):
        np.random.seed(42)
        y_true = np.eye(3)
        y_true = y_true.reshape((1,3,3,1))
        y_pred = y_true
        dc = LogCoshDiceError()
        loss = dc(y_true,y_pred)

        self.assertTrue(loss < 1e-5)

    def test_lchdice_error_max(self):

        np.random.seed(42)

        y_true = np.eye(3).reshape((1,3,3,1))
        y_pred = np.zeros((1,3,3,1))

        dc = LogCoshDiceError(1)
        loss = dc(y_true,y_pred)

        self.assertTrue(loss > 0.24)

if __name__ == '__main__':
    unittest.main()