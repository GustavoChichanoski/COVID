import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python import keras
from tensorflow import keras
from pathlib import Path

from src.models.classificacao.funcional_model import classification_model

from src.data.classification.cla_dataset import Dataset
from src.data.classification.cla_generator import (
    ClassificationDatasetGenerator as ClaDataGen,
)
from src.output_result.folders import remove_folder, zip_folder
from src.models.grad_cam_split import grad_cam, last_act_after_conv_layer
from src.plots.graph import plot_dataset
from src.models.classificacao.funcional_model import (
    base,
    classification_model,
    get_callbacks,
    get_classifier_layer_names,
    make_grad_cam,
    confusion_matrix,
    save_weights
)
from src.plots.evaluation_classification import plot_mc_in_csv

from tensorflow.python.keras.callbacks import (
    Callback,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
    EarlyStopping
)

DIM_ORIGINAL = 1024
DIM_SPLIT = 224
CHANNELS = 1
SHAPE = (DIM_SPLIT, DIM_SPLIT, CHANNELS)
K_SPLIT = 100
BATCH_SIZE = 32
EPOCHS = 100
DATA = Path('data')
TEST = 'data/0000.png'
TAMANHO = 0
LR = 1e-4
WEIGHTS = ['resnet.best.weights.hdf5', 'inception.best.weights.hdf5', 'vgg.best.weights.hdf5', 'best.weights.hdf5']
REDES = ["VGG16", "ResNet50V2", "InceptionResNetV2", "VGG19", "DenseNet121"]
LABELS = ["Covid", "Normal", "Pneumonia"]

rede = 'ResNet101V2'
hist = pd.read_csv(f'figs/{rede}/history_{rede}.csv')

# plt.plot(hist['lr'])
# plt.xlabel('Epochs')
# plt.ylabel('Learning rate')
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.grid('on')
# plt.title(f'Learning rate to {rede} along the epochs')
# plt.savefig(f"figs/{rede}/lr_{rede}.png", dpi=600)
# plt.show()

# max_loss = hist['loss'].min()
# max_val_loss = hist['val_loss'].min()
# plt.plot(hist['loss'])
# plt.plot(hist['val_loss'])
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid('on')
# plt.title(f'Loss to model {rede} along the epochs')
# plt.legend([f'train: {max_loss:.4f}', f'validation: {max_val_loss:.4f}'])
# plt.savefig(f"figs/{rede}/loss_{rede}.png", dpi=600)
# plt.show()

# max_acc = hist['accuracy'].max()
# max_val_acc = hist['val_accuracy'].max()
# plt.plot(hist['accuracy'])
# plt.plot(hist['val_accuracy'])
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.grid('on')
# plt.title(f'Accuracy to model {rede} along the epochs')
# plt.legend([f'train: {max_acc:.4f}', f'validation: {max_val_acc:.4f}'])
# plt.savefig(f"figs/{rede}/acc_{rede}.png", dpi=600)
# plt.show()

np.random.seed(0)

TRAIN_PATH = DATA / "train"
TEST_PATH = DATA / "test"
LABELS = ["Covid", "Normal", "Pneumonia"]
model = classification_model(
    DIM_SPLIT,
    channels=CHANNELS,
    classes=len(LABELS),
    drop_rate=0,
    model_name=rede
)
model.compile(
    loss="binary_crossentropy",
    optimizer=Adamax(learning_rate=LR),
    metrics="accuracy"
)
model.summary()

winner = make_grad_cam(model=model,
                       image= DATA / 'Covid' / '0000.png',
                       n_splits=400,
                       threshold=0.1)
