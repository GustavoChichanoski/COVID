from src.models.grad_cam_split import grad_cam
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback
from src.models.classificacao.funcional_model import (
    base,
    get_callbacks,
    get_classifier_layer_names,
    last_act_after_conv_layer,
    last_conv_layer,
)
from tensorflow.python import keras
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class TestFuncionalModel(unittest.TestCase):
    def test_base(self) -> None:
        base_model = base()
        self.assertTrue(isinstance(base_model, Model))

    def test_get_callbacks(self) -> None:
        callbacks = get_callbacks()
        for i in range(len(callbacks)):
            self.assertTrue(isinstance(callbacks[i], Callback))

    def test_last_conv_layer(self) -> None:

        model_builder = keras.applications.xception.Xception

        last_conv_layer_name = "block14_sepconv2_act"

        # Make model
        model = model_builder(weights="imagenet")

        # Remove last layer's softmax
        model.layers[-1].activation = None

        classification_layers_names = last_act_after_conv_layer(model)

        self.assertEqual(last_conv_layer_name, classification_layers_names)

    def test_classifier_layer_names(self):
        
        model_builder = keras.applications.xception.Xception

        last_conv_layer_name = "block14_sepconv2_act"

        model = model_builder(weights="imagenet")

        model.layers[-1].activation = None

        classification_layers_names = get_classifier_layer_names(model, last_conv_layer_name)

        self.assertEqual(['predictions', 'avg_pool'],classification_layers_names)

    def test_grad_cam(self) -> None:

        img_size = (299, 299)
        model_builder = keras.applications.xception.Xception
        preprocess_input = keras.applications.xception.preprocess_input
        decode_predictions = keras.applications.xception.decode_predictions

        last_conv_layer_name = "block14_sepconv2_act"

        # The local path to our target image
        img_path = keras.utils.get_file(
            "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
        )

        # Prepare image
        img_array = preprocess_input(get_img_array(img_path, size=img_size))

        # Make model
        model = model_builder(weights="imagenet")

        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Print what the top predicted class is
        preds = model.predict(img_array)
        print("Predicted:", decode_predictions(preds, top=1)[0])

        last_act_name = last_act_after_conv_layer(model)
        classification_layers_names = get_classifier_layer_names(model, last_conv_layer_name)

        # Generate class activation heatmap
        heatmap = grad_cam(
            image=img_array,
            model=model,
            classifier_layer_names=classification_layers_names,
            last_conv_layer_name=last_act_name
        )

        # Display heatmap
        plt.matshow(heatmap)
        plt.show()


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array