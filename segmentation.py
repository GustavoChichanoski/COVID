from pathlib import Path
import scipy.ndimage as ndimage
import pandas as pd
import numpy as np
import segmentation_models as sm
import tensorflow as tf
import matplotlib.pyplot as plt

from src.data.segmentation.data_seg import (
    parse_image, load_image_train, load_image_test
)

BACKBONE = 'resnet34'
BATCH_SIZE = 16
EPOCHS = 150
LERNING_RATE = 5e-2
IMG_SIZE = 256
BETA = 0.25
ALPHA = 0.25
GAMMA = 2
SMOOTH = 1

# %% load data
df_dataset = pd.read_csv('dataset\metadata_segmentation.csv')

df_dataset['lung'] = df_dataset['lung'].apply(lambda x: str(Path.cwd() / x))
df_dataset['mask'] = df_dataset['mask'].apply(lambda x: str(Path.cwd() / x))

df_train = df_dataset.where(df_dataset['type'] == 'train').dropna()
df_valid = df_dataset.where(df_dataset['type'] == 'valid').dropna()
df_tests = df_dataset.where(df_dataset['type'] == 'tests').dropna()

# Define Constants

TRAIN_LENGTH = len(df_train)
VALID_LENGTH = len(df_valid)
TESTS_LENGTH = len(df_tests)

BATCH_SIZE = 16
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# %% code
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")

# %% fixed random
SEED = 42

# %% find gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUS, ", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

# %% Creating dataset
# Image size
IMG_SIZE = 256
NCHANNELS = 3
N_CLASSES = 1

# %% Creating source dataset
train_paths = tf.data.Dataset.list_files(df_train['lung'].values, seed=SEED)
valid_paths = tf.data.Dataset.list_files(df_valid['lung'].values, seed=SEED)
tests_paths = tf.data.Dataset.list_files(df_tests['lung'].values, seed=SEED)

dataset_path = {
    'train': train_paths,
    'valid': valid_paths,
    'tests': tests_paths
}

# %%
dataset_path['train'] = dataset_path['train'].map(parse_image, num_parallel_calls=AUTOTUNE)
dataset_path['train'] = dataset_path['train'].map(load_image_train, num_parallel_calls=AUTOTUNE)
dataset_path['train'] = dataset_path['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
dataset_path['train'] = dataset_path['train'].batch(BATCH_SIZE)
dataset_path['train'] = dataset_path['train'].prefetch(buffer_size=AUTOTUNE)

dataset_path['valid'] = dataset_path['valid'].map(parse_image)
dataset_path['valid'] = dataset_path['valid'].map(load_image_test)
dataset_path['valid'] = dataset_path['valid'].batch(BATCH_SIZE)
dataset_path['valid'] = dataset_path['valid'].prefetch(buffer_size=AUTOTUNE)

dataset_path['tests'] = dataset_path['tests'].map(parse_image)
dataset_path['tests'] = dataset_path['tests'].map(load_image_test)
dataset_path['tests'] = dataset_path['tests'].batch(BATCH_SIZE)
dataset_path['tests'] = dataset_path['tests'].prefetch(buffer_size=AUTOTUNE)

def plot_batch_sizes(ds):
  batch_sizes = [batch.shape[0] for batch in ds]
  plt.bar(range(len(batch_sizes)), batch_sizes)
  plt.xlabel('Batch number')
  plt.ylabel('Batch size')

def random_rotate_image(image, mask):
  angle = np.random.uniform(-30, 30)
  image = ndimage.rotate(image, angle, reshape=False)
  mask = ndimage.rotate(mask, angle, reshape=False)
  return image, mask

def show(image, mask):
  plt.figure()
  plt.subplot(2,1,1)
  plt.imshow(image[0,:,:,0])
  plt.title('pulmao')
  plt.subplot(2,1,2)
  plt.imshow(mask[0,:,:,0])
  plt.title('mascara')
  plt.axis('off')

for image, mask in dataset_path['tests'].take(2):
  image, mask = random_rotate_image(image, mask)
  show(image, mask)

# %% create model
sm.set_framework('tf.keras')
sm.framework()

preprocess_input = sm.get_preprocessing(BACKBONE)

dataset_path['train'] = preprocess_input(dataset_path['train'])
dataset_path['valid'] = preprocess_input(dataset_path['valid'])
dataset_path['tests'] = preprocess_input(dataset_path['tests'])

input_layer = tf.keras.Input((256,256, 3))
layer = tf.keras.layers.RandomFlip("horizontal", seed=SEED)(input_layer)
layer = tf.keras.layers.RandomZoom(
        height_factor=(-0.1, -0.1),
        width_factor=(-0.1, -0.1),
        seed=SEED) (layer)
layer = tf.keras.layers.RandomRotation(0.3)(layer)

layer = sm.Unet(
    BACKBONE,
    encoder_weights='imagenet',
    classes=1,
    activation='sigmoid'
  ) (layer)
model = tf.keras.Model(inputs=input_layer, outputs=layer)

model.compile(
  'Adam',
  sm.losses.bce_jaccard_loss,
  metrics=[sm.metrics.iou_score]
)

model.fit(
  x=dataset_path['train'],
  batch_size=BATCH_SIZE,
  epochs=2,
  validation_data=dataset_path['valid']
)

sm.utils.set_trainable(model)
model.fit()
