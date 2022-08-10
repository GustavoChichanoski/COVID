from tensorflow.python.keras.callbacks import (
    Callback,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
    EarlyStopping
)
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import cv2

from src.images.process_images import split_images_n_times as split_n_time
from src.models.classificacao.funcional_model import (
    classification_model, model_compile)
from src.data.classification.cla_dataset_csv import DatasetCsv
from src.data.classification.cla_generator import ClassificationDatasetGenerator as ClaDataGen


def read_image(path: Path):
    lung = cv2.imread(str(path))
    return lung


dataset = pd.read_csv(Path.cwd() / 'data/metadata.csv')
for key in dataset.columns:
    print(key)
    if 'type' in key or 'survival' in key or 'Unnamed' in key:
        continue
    dataset[key] = dataset[key].apply(
        lambda x: str(Path.cwd() / x).replace('\\', '/'))

model_classification_params = {
    'dim': 224,
    'channels': 1,
    'classes': 3,
    'final_activation': 'softmax',
    'activation': 'relu',
    'drop_rate': 0.2,
    'model_name': 'ResNet50V2',
}
class_model = classification_model(**model_classification_params)
model_compile(class_model)

class_dataset = DatasetCsv(dataset)
class_dataset.set_labels()
part_param = {"tamanho": 0}
train, validation, tests = class_dataset.partition(
    val_size=0.2,
    test_size=0.1,
    **part_param
)

params = {
    'dim': 224,
    'batch_size': 32,
    'n_class': 3,
    'channels': 1,
    'threshold': 0.1,
    'desire_len': None
}
train_generator = ClaDataGen(train[0], train[1], **params)
val_generator = ClaDataGen(validation[0], validation[1], **params)
test_generator = ClaDataGen(tests[0], tests[1], **params)

callbacks = []
check_params = {
    "monitor": "val_loss",
    "verbose": 0,
    "mode": "min",
    "save_best_only": True,
    "save_weights_only": True,
}
callbacks.append(ModelCheckpoint(".\\best.weights.hdf5", **check_params))

# Reduz o valor de LR caso o monitor nao diminuia
reduce_params = {
    "factor": 0.5,
    "patience": 3,
    "verbose": 1,
    "mode": "min",
    "min_delta": 1e-3,
    "cooldown": 5,
    "min_lr": 1e-8,
}
callbacks.append(ReduceLROnPlateau(monitor="val_loss", **reduce_params))

callbacks.append(EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
))

history = class_model.fit(
    x=train_generator,
    validation_data=val_generator,
    epochs=1,
    batch_size=32,
    callbacks=callbacks
)
