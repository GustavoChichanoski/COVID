from os import path
from pathlib import Path
from src.plots.evaluation_classification import plot_mc_in_csv
from src.plots.graph import plot_dataset, test
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
    confusion_matrix,
    get_callbacks,
    get_classifier_layer_names,
    make_grad_cam,
    save_weights,
)

DIM_ORIGINAL = 1024
DIM_SPLIT = 224
CHANNELS = 1
K_SPLIT = 400
BATCH_SIZE = 1
EPOCHS = 2
TAMANHO = 0

DATA = Path("D:\\Mestrado") / "datasets" / "new_data"
TRAIN_PATH = DATA / "train"
TEST_PATH = DATA / "test"
LABELS = ["Covid", "Normal", "Pneumonia"]

REDES = ["ResNet50V2", "InceptionResNetV2", "VGG19", "DenseNet121"]
NET = REDES[0]
ds_train = Dataset(path_data=TRAIN_PATH, train=False)
ds_test = Dataset(path_data=TEST_PATH, train=False)

# fixa a aleatoriedade do numpy random
np.random.seed(0)
part_param = {"tamanho": TAMANHO, "shuffle": False}
train, validation = ds_train.partition(val_size=0.2, **part_param)
test_values, _test_val_v = ds_test.partition(val_size=1e-3, **part_param)

model = classification_model(DIM_SPLIT, channels=1, classes=len(LABELS), model_name=NET)
model.compile(
    loss="binary_crossentropy",
    optimizer=Adamax(learning_rate=1e-5),
    metrics="accuracy",
)
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

# history = model.fit(
#     x=train_generator,
#     validation_data=val_generator,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     callbacks=callbacks,
# )

# save_weights(
#     modelname="resnet",
#     model=model,
#     history=history,
# )

# print("Make Grad Cam")


output = Path("outputs") / NET

model.load_weights(output / "weights\\best.weights.hdf5")

# winner = make_grad_cam(
#     model=model,
#     image=test_generator.x[0],
#     n_splits=K_SPLIT,
#     threshold=0.1,
#     orig_dim=DIM_ORIGINAL,
# )
matriz = confusion_matrix(model,test_generator,1)

# matriz = np.array([[267,3,1],[1,502,27],[14,46,845]])

# import matplotlib

plot_dataset(matriz,K_SPLIT,path=output / "figures",pgf=False)

parar = True

# zip_folder(Path.cwd())

# remove_folder('./Covid')

# assert_equal(winner, "Covid")