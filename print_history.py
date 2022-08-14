import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from pathlib import Path

from src.models.classificacao.funcional_model import classification_model

from src.models.classificacao.funcional_model import (
    classification_model,
    make_grad_cam
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
REDES = ["DenseNet169", "VGG16", "ResNet50V2", "InceptionResNetV2", "VGG19", "DenseNet121"]
LABELS = ["Covid", "Normal", "Pneumonia"]

rede = 'DenseNet169'
# hist = pd.read_csv(f'figs/{rede}/history_{rede}.csv')

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
                       image= DATA / 'Covid' / '1720.png',
                       n_splits=10,
                       threshold=0.35,
                       name=f'figs/{rede}/grad_{rede}.png')

print('agua')
