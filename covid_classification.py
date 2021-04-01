# %% [code]
"""
    O Objetivo desse programa é classificar raios-x dos pulmões entre pacientes
    normais, com pneumonia ou com covid-19
"""
from src.output_result.folders import create_folders, remove_folder
from src.plots.graph import plot_dataset
from src.model.model import ModelCovid
from src.dataset.dataset import Dataset
from src.model.generator import DataGenerator
from pathlib import Path
from os import listdir
import tensorflow as tf
import sys
import numpy as np

# %% [code]

DIM_ORIGINAL = 1024
DIM_SPLIT = 224
K_SPLIT = 100
BATCH_SIZE = 1
EPOCHS = 1

NETS = ['DenseNet201',
        'InceptionResNetV2',
        'ResNet50V2',
        'VGG19']

__version__ = '1.0'

# %% [code]
KAGGLE = False
system = Path('../input')

if system.exists():
    __SYS = system / listdir('../input')[0]
    sys.path.append(str(__SYS))
    KAGGLE = True
else:
    __SYS = Path('./')

del system

# %% [code] Paths
DATA = __SYS / 'data'
TRAIN_PATH = DATA / 'train'
TEST_PATH = DATA / 'test'
TEST = TEST_PATH / 'Covid/0000.png'
CWD = Path.cwd()
OUTPUT_PATH = CWD / 'outputs'
CLEAR = True

if CLEAR:
    remove_folder([OUTPUT_PATH, Path('./logs'), Path('./build')])

nets_path = create_folders(name=OUTPUT_PATH, nets=NETS)

# %% Initialize TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync
except:
    tpu = None
# %% [code]

np.random.seed(seed=42)

labels = listdir(TRAIN_PATH)

dataset = Dataset(path_data=TRAIN_PATH)
test = Dataset(path_data=TEST_PATH)

train, validation = dataset.partition(.9992)
test_values, _test_val_v = test.partition(1e-5)

params = {'dim': DIM_SPLIT, 'batch_size': BATCH_SIZE,
          'n_class': len(labels), 'shuffle': True, 'channels': 3}
train_generator = DataGenerator(data=train, **params)
val_generator = DataGenerator(data=validation, **params)
test_generator = DataGenerator(data=test_values, **params)

# Detect and init TPU
# %% [code]
for model, net_path in zip(NETS, nets_path):

    model_params = {'labels': labels, 'model': model,
                    'model_input_shape': (DIM_SPLIT, DIM_SPLIT, 3)}
    if tpu is not None:
        with tpu_strategy.scope():
            covid = ModelCovid(weight_path='.model/weightss.best.hfd5', **model_params)
            covid.compile(loss='categorical_crossentropy', lr=1e-5,steps_per_execution=32)
    else:
        covid = ModelCovid(weight_path='.model/weightss.best.hfd5', **model_params)
        covid.compile(loss='categorical_crossentropy', lr=1e-5)

    path_weight = net_path / 'weights'
    path_figure = net_path / 'figures'

    weight = None
    for weight in path_weight.iterdir():
        if weight.suffix == 'hdf5':
            break
    if weight is not None:
        covid.load(weight)
    else:
        fit_params = {'epochs': EPOCHS, 'shuffle': True,
                      'workers': 1, 'batch_size': BATCH_SIZE}
        
        history = covid.fit_generator(
            train_generator=train_generator,
            val_generator=None, **fit_params
        )

        weight = covid.save(path=net_path, model=model,
                            history=history.history, metric='accuracy')

    name = path_figure / '{}_{}'.format(model, K_SPLIT)
    covid.predict(image=TEST, n_splits=K_SPLIT, name=name, grad=False)

    matrix = covid.confusion_matrix(test_generator.x, 1)
    plot_dataset(names=labels, absolut=matrix, n_images=1, path=path_figure)
