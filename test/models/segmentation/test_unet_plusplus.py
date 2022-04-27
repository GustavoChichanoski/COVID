from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.models import Input, Model
from tensorflow.python.keras.layers import BatchNormalization
from src.models.segmentation.unet_plus_plus import *

import unittest


class TestUnetPlusPlus(unittest.TestCase):
    def test_double_values_n_times(self) -> None:
        double_values = double_values_n_times(initial_value=5, array_size=3)
        self.assertTrue(isinstance(double_values, list))
        self.assertEqual(double_values, [5, 10, 20])

    def test_conv_plus_plus(self) -> None:
        matrix = (0, 0)
        shape = (256, 256, 1)
        inputs = Input(shape)
        outputs = conv_plus_plus(
            layer=inputs,
            filters=32,
            kernel_size=3,
            activation="relu",
            matrix_position=matrix,
        )
        model = Model(inputs=inputs, outputs=outputs)
        self.assertTrue(len(model.layers) > 2, "A convolução não foi gerada")
        for layer in model.layers[1:]:
            if isinstance(layer, BatchNormalization):
                self.assertEqual(layer.name, f"bn_{matrix[0]}_{matrix[1]}")
            elif isinstance(layer, Activation):
                self.assertEqual(layer.name, f"act_{matrix[0]}_{matrix[1]}")
            elif isinstance(layer, Conv2D):
                self.assertEqual(layer.name, f"cv_{matrix[0]}_{matrix[1]}")

    def test_unet_plusplus(self) -> None:
        model = unet_plus_plus(depth=5)
        self.assertTrue(isinstance(model, Model))
        count_conv = 0
        count_act = 0
        count_bn = 0
        for layer in model.layers:
            if isinstance(layer, Conv2D):
                count_conv += 1
            if isinstance(layer, Activation):
                count_act += 1
            if isinstance(layer, BatchNormalization):
                count_bn += 1
        self.assertEqual(count_conv, 16)
        self.assertEqual(count_act, 16)
        self.assertEqual(count_bn, 16)

    def test_backbone(self) -> None:
        depth = 5
        shape = (256, 256, 1)
        filters = 8
        kernel_size = 3
        activation = "relu"
        inputs = Input(shape)
        outputs = backbone(
            layer=inputs,
            depth=depth,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
        )
        model = Model(inputs=inputs, outputs=outputs)

        self.assertEqual(
            len(model.layers),
            1 + depth * 4,
            f"O modelo era para gerar {1 + depth * 4}, mas gerou {len(model.layers)}",
        )


if __name__ == "__main__":
    unittest.main()