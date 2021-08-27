from src.models.losses.dice_loss import DiceError
from src.models.segmentation.unet import Unet
from tensorflow.python.keras.models import Model
from tensorflow import GradientTape
import unittest

class TestSegModel(unittest.TestCase):

    def test_model_segmentation(self):

        loss = DiceError()

        model = Unet()
        self.assertTrue(model is not None)
        model.compile()
        self.assertTrue(len(model.layers) > 0)



if __name__ == '__main__':
    unittest.main()