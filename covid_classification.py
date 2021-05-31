# %% [code]
"""
    O Objetivo desse programa é classificar raios-x dos pulmões entre pacientes
    normais, com pneumonia ou com covid-19
"""
from src.plots.history import plot_history
from src.plots.graph import plot_dataset
from src.model.model import ModelCovid
from src.dataset.dataset import Dataset
from src.dataset.generator import DataGenerator
from src.output_result.folders import *
from zipfile import ZipFile
from pathlib import Path
from os import listdir
import sys
import numpy as np
import cProfile
# %% [code]

DIM_ORIGINAL = 1024
DIM_SPLIT = 224
CHANNELS = 1
SHAPE = (DIM_SPLIT, DIM_SPLIT, CHANNELS)
K_SPLIT = 100
BATCH_SIZE = 1
EPOCHS = 2

NETS = ["DenseNet201", "InceptionResNetV2", "ResNet50V2", "VGG19"]

__version__ = "1.0"

# %% [code]
KAGGLE = False
system = Path("../input")

if system.exists():
    __SYS = system / listdir("../input")[0]
    sys.path.append(str(__SYS))
    KAGGLE = True
else:
    __SYS = Path("./")

del system

# %% [code] Paths
DATA = __SYS / "data"
TRAIN_PATH = DATA / "train"
TEST_PATH = DATA / "test"
TEST = TEST_PATH / "Covid/0000.png"
CWD = Path.cwd()
OUTPUT_PATH = CWD / "outputs"
CLEAR = False
LABELS = ["Covid", "Normal", "Pneumonia"]
if CLEAR:
    remove_folder([OUTPUT_PATH, Path("./logs"), Path("./build")])

nets_path = create_folders(name=OUTPUT_PATH, nets=NETS)
# %% [code]
# np.random.seed(seed=42)
labels = listdir(TRAIN_PATH)

ds_train = Dataset(path_data=TRAIN_PATH, train=False)
ds_test = Dataset(path_data=TEST_PATH, train=False)

part_param = {"tamanho": 0}
train, validation = ds_train.partition(val_size=0.2, **part_param)
test_values, _test_val_v = ds_test.partition(val_size=1e-5, **part_param)

params = {
    "dim": DIM_SPLIT,
    "batch_size": BATCH_SIZE,
    "n_class": len(labels),
    "channels": CHANNELS,
}
train_generator = DataGenerator(x_set=train[0], y_set=train[1], **params)
val_generator = DataGenerator(x_set=validation[0], y_set=validation[1], **params)
test_generator = DataGenerator(x_set=test_values[0], y_set=test_values[1], **params)
# %% [code]
for net, net_path in zip(NETS[1:], nets_path[1:]):
    
    model = net
    net_path = net_path

    model_params = {"labels": labels, "model_name": model, "model_input_shape": SHAPE}
    covid = ModelCovid(weight_path=".model/weights.best.hfd5", **model_params)
    covid.compile(loss="categorical_crossentropy", lr=1e-5)

    path_weight = net_path / "weights"
    path_figure = net_path / "figures"
    path_history = net_path / "history"

    weight = last_file(path_weight)

    if weight is not None:
        print(f"[INFO] Carregando o modelo: {weight}")
        covid.load(weight)
    else:
        fit_params = {
            "epochs": EPOCHS,
            "shuffle": True,
            "workers": 1,
            "batch_size": BATCH_SIZE,
            "verbose": True,
        }
        history = covid.fit_generator(
            train_generator=train_generator, val_generator=val_generator, **fit_params
        )
        file_model, file_weights, file_history = covid.save(
            path=net_path, history=history.history, metric="accuracy", kaggle=KAGGLE
        )
        plot_history(history)

    covid.model.summary()

    name = path_figure / f"{model}_{K_SPLIT}"
    print(f"[INFO] Predição de uma imagem: {K_SPLIT}")
    print(covid.predict(image=TEST, n_splits=K_SPLIT, name=name, grad=False,threshold=0.75))

    # matrix = covid.confusion_matrix(test_generator.x, 4)
    # plot_dataset(absolut=matrix, names=labels, n_images=1, path=path_figure)

zip_folder(OUTPUT_PATH)