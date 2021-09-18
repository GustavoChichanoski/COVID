from pathlib import Path
from src.output_result.folders import remove_folder, zip_folder

from numpy.testing._private.utils import assert_equal
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python import keras
from tensorflow import keras

from src.dataset.classification.cla_dataset import Dataset
from src.dataset.classification.cla_generator import (
    ClassificationDatasetGenerator as ClaDataGen,
)
from src.models.grad_cam_split import grad_cam, last_act_after_conv_layer
from src.models.classificacao.funcional_model import (
    base,
    classification_model,
    get_callbacks,
    get_classifier_layer_names,
    make_grad_cam,
    save_weights,
)

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

    def test_classifier_layer_names(self) -> None:

        model_builder = keras.applications.xception.Xception

        last_conv_layer_name = "block14_sepconv2_act"

        model = model_builder(weights="imagenet")

        model.layers[-1].activation = None

        classification_layers_names = get_classifier_layer_names(
            model, last_conv_layer_name
        )

        self.assertEqual(["avg_pool", "predictions"], classification_layers_names)

    def teste_model(self) -> None:

        np.random.seed(20)
        DIM_ORIGINAL = 1024
        DIM_SPLIT = 224
        CHANNELS = 1
        K_SPLIT = 10
        BATCH_SIZE = 1
        EPOCHS = 2

        DATA = Path("D:\\Mestrado") / "datasets" / "new_data"
        TRAIN_PATH = DATA / "train"
        TEST_PATH = DATA / "test"
        LABELS = ["Covid", "Normal", "Pneumonia"]

        ds_train = Dataset(path_data=TRAIN_PATH, train=False)
        ds_test = Dataset(path_data=TEST_PATH, train=False)

        part_param = {"tamanho": 10, "shuffle": False}
        train, validation = ds_train.partition(val_size=0.2, **part_param)
        test_values, _test_val_v = ds_test.partition(val_size=1e-3, **part_param)

        model = classification_model(DIM_SPLIT, channels=1, classes=len(LABELS))
        model.compile(loss="categorical_crossentropy", optimizer=Adamax(learning_rate=1e-5), metrics="accuracy")
        model.summary()

        params = {
            "dim": DIM_SPLIT,
            "batch_size": BATCH_SIZE,
            "n_class": len(LABELS),
            "channels": CHANNELS,
            "threshold": 0.25,
        }
        train_generator = ClaDataGen(train[0], train[1], **params)
        val_generator = ClaDataGen(validation[0], validation[1], **params)
        test_generator = ClaDataGen(test_values[0], test_values[1], **params)

        callbacks = get_callbacks()

        history = model.fit(
            x=train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks
        )

        save_weights(
            modelname='resnet',
            model=model,
            history=history,
        )

        zip_folder(Path.cwd())

        remove_folder('./Covid')

        print("Make Grad Cam")

        winner = make_grad_cam(
            model=model,
            image=test_generator.x[0],
            n_splits=K_SPLIT,
            threshold=0.1,
            orig_dim=DIM_ORIGINAL
        )
        
        assert_equal(winner, 'Covid')

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
        classification_layers_names = get_classifier_layer_names(
            model, last_conv_layer_name
        )

        # Generate class activation heatmap
        heatmap = grad_cam(
            image=img_array,
            model=model,
            classifier_layer_names=classification_layers_names,
            last_conv_layer_name=last_act_name,
        )

        heatmap_expected = np.array(
            [
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    1.679093111306428909e-03,
                    6.725340150296688080e-03,
                    1.920665055513381958e-02,
                    2.625155262649059296e-02,
                    7.416199892759323120e-03,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                ],
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    6.788050290197134018e-03,
                    7.864440232515335083e-02,
                    1.785268336534500122e-01,
                    3.584012985229492188e-01,
                    4.312762022018432617e-01,
                    3.502597510814666748e-01,
                    5.523559451103210449e-02,
                    2.611180243548005819e-04,
                ],
                [
                    2.508753095753490925e-04,
                    6.195336580276489258e-03,
                    1.466881781816482544e-01,
                    3.973096013069152832e-01,
                    6.196283102035522461e-01,
                    8.991926312446594238e-01,
                    9.291224479675292969e-01,
                    7.991613149642944336e-01,
                    3.614846765995025635e-01,
                    1.031120121479034424e-01,
                ],
                [
                    9.424575604498386383e-03,
                    9.077487140893936157e-02,
                    4.009900689125061035e-01,
                    6.209744215011596680e-01,
                    8.102996945381164551e-01,
                    1.000000000000000000e00,
                    9.777907729148864746e-01,
                    8.824282288551330566e-01,
                    5.160366892814636230e-01,
                    2.537339329719543457e-01,
                ],
                [
                    1.977579109370708466e-02,
                    1.387653648853302002e-01,
                    4.493062794208526611e-01,
                    6.306136846542358398e-01,
                    6.956336498260498047e-01,
                    7.861608862876892090e-01,
                    6.660469770431518555e-01,
                    5.836380720138549805e-01,
                    2.786544263362884521e-01,
                    1.065999716520309448e-01,
                ],
                [
                    5.166627932339906693e-03,
                    6.239785254001617432e-02,
                    3.070983886718750000e-01,
                    4.491437971591949463e-01,
                    3.987758457660675049e-01,
                    2.670759260654449463e-01,
                    1.261870712041854858e-01,
                    5.646723136305809021e-02,
                    1.006829366087913513e-02,
                    1.239834236912429333e-03,
                ],
                [
                    6.551821570610627532e-06,
                    1.406515482813119888e-03,
                    3.027859702706336975e-02,
                    1.004303023219108582e-01,
                    8.144357800483703613e-02,
                    2.835766412317752838e-02,
                    2.737278817221522331e-03,
                    2.417014213278889656e-03,
                    9.001792641356587410e-04,
                    1.974020706256851554e-04,
                ],
                [
                    0.000000000000000000e00,
                    2.621802559588104486e-04,
                    8.406823035329580307e-04,
                    3.146168543025851250e-03,
                    4.274466075003147125e-03,
                    3.943377640098333359e-03,
                    3.003499237820506096e-03,
                    2.137763891369104385e-03,
                    8.062763954512774944e-04,
                    8.731170964892953634e-05,
                ],
                [
                    0.000000000000000000e00,
                    2.300558990100398660e-04,
                    4.016221209894865751e-04,
                    1.068474026396870613e-03,
                    1.713583362288773060e-03,
                    1.966057112440466881e-03,
                    1.153069082647562027e-03,
                    3.318891394883394241e-04,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                ],
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    2.398468132014386356e-05,
                    2.686489606276154518e-04,
                    4.752037639264017344e-04,
                    3.999800246674567461e-04,
                    2.011069882428273559e-04,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                ],
            ]
        )

        diff = np.abs(heatmap) - np.abs(heatmap_expected) < 0.5
        diff = np.sum(diff)

        self.assertEquals(100, diff)


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array