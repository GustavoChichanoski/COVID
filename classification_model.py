# %%
from typing import List
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import segmentation_models as sm
import albumentations as A

from pathlib import Path
from tensorflow.keras import Model
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tqdm import tqdm

from src.models.grad_cam_split import prob_grad_cam
from src.models.losses.log_cosh_dice_loss import LogCoshDiceError
from src.models.classificacao.funcional_model import (
    classification_model, plot_gradcam
)
from src.images.process_images import split_images_n_times
from src.images.read_image import read_images as ri

SPLIT = 224
CHANNELS = 1
THRESHOLD = 0.5
N_SPLITS = 400
DIM_ORIGINAL = 1024
DIM_SPLIT = 224
CHANNELS = 1
SHAPE = (DIM_SPLIT, DIM_SPLIT, CHANNELS)
K_SPLIT = 400
BATCH_SIZE = 32
EPOCHS = 100
DATA = Path('./data')
TEST = Path('./data/test/Pneumonia/0838.png')
TAMANHO = 0
LR = 1e-4
WEIGHTS = [
    'resnet.best.weights.hdf5', 'inception.best.weights.hdf5',
    'vgg.best.weights.hdf5', 'best.weights.hdf5'
]
REDES = [
    "ResNet50V2",
    "InceptionResNetV2",
    "VGG19",
    "DenseNet121",
]
PESOS = [
    'pesos\\resnet.best.weights.hdf5',
    'pesos\\inception.best.weights.hdf5',
    'pesos\\vgg.best.weights.hdf5',
    'pesos\\best.weights.hdf5',
]
TRAIN_PATH = DATA / "train"
TEST_PATH = DATA / "test"
LABELS = ["Covid", "Normal", "Pneumonia"]
COLUNM_NAMES = [
    'original', 'mask', 'segmentation', 'gradCAM_Resnet', 'gradCAM_Inception',
    'gradCAM_VGG', 'gradCAM_DenseNet', 'lungGradCAM_Resnet',
    'lungGradCAM_Inception', 'lungGradCAM_VGG', 'lungGradCAM_DenseNet', 'type',
    'survival'
]
UNET_PARAMS = {
    'input_shape': (None, None, 3),
    'encoder_freeze': False,
    'encoder_weights': None,
    'classes': 1,
    'activation': 'sigmoid'
}
BACKBONE = 'resnet34'
LERNING_RATE = 5e-2
IMG_SIZE = 256
IMAGE_NAME = '{:04}.png'


# %%
def create_segmentation_model(backbone: str, unet_params,
                              dim_input: int) -> Model:
    input_layer = tf.keras.Input((dim_input, dim_input, 1))
    layer = tf.keras.layers.Conv2D(filters=3,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu')(input_layer)
    layer = sm.Unet(backbone, **unet_params)(layer)
    model = tf.keras.Model(inputs=input_layer, outputs=layer)
    return model


def segmentation_model_compile(model: Model) -> None:
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    log_dice = LogCoshDiceError(regularization_factor=100)
    model.compile(opt,
                  loss=log_dice,
                  metrics=[sm.metrics.IOUScore(),
                           sm.metrics.FScore()])


def remove_dump_colunms(csv_file: str, dataset_ieee: Path) -> pd.DataFrame:
    df_ieee = pd.read_csv(dataset_ieee / csv_file)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['finding'] == 'todo'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['view'] == 'L'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['view'] == 'Axial'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['finding'] == 'todo'].index)
    return df_ieee


def save_image(lung, path: Path, i: int) -> None:
    lung_path = path / IMAGE_NAME.format(i)
    cv2.imwrite(str(lung_path), lung)


# %% function
rede = REDES[0]
ieee = Path('ieee_dataset')
ieee_images = pd.read_csv(ieee / 'metadata.csv')
ieee_annotations = pd.read_csv(ieee / 'annotations' /
                               'covid-severity-scores.csv')
ieee_images = remove_dump_colunms('metadata.csv', ieee)

# %%
dataset = Path('data')
orig = dataset / 'original'
mask = dataset / 'mask'
seg = dataset / 'segmentation'
grad = dataset / 'gradCAM'
lungGradCAM = dataset / 'lungGradCAM'
# %% segmentation model
model_seg = create_segmentation_model(BACKBONE, UNET_PARAMS, IMG_SIZE)
segmentation_model_compile(model_seg)
model_seg.load_weights('.\\kaggle\\model.hdf5')
# %% classification model
models = []
for rede, peso in zip(REDES, PESOS):
    model = classification_model(DIM_SPLIT,
                                 channels=CHANNELS,
                                 classes=len(LABELS),
                                 drop_rate=0,
                                 model_name=rede)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adamax(learning_rate=LR),
                  metrics="accuracy")
    model.load_weights(peso)
    models.append(model)
# %% split image
params_splits = {
    'verbose': True,
    'dim_split': SPLIT,
    'threshold': THRESHOLD,
    'n_split': N_SPLITS
}


# %%
def compose_original(height: int = 1024, width=1024) -> A.Compose:
    return A.Compose([
        A.Resize(height=height, width=width, always_apply=True),
        A.Equalize(always_apply=True),
        A.InvertImg(always_apply=True)
    ])


def save_image_segmentation(i: int,
                            model_seg: Model,
                            image_norm,
                            image,
                            dim_spit: int = 256,
                            dim_orig: int = 1024):
    image_model_seg = cv2.resize(image_norm, (dim_spit, dim_spit))
    image_model_seg = image_model_seg.reshape((1, dim_spit, dim_spit, 1))
    mask = model_seg.predict(image_model_seg)
    mask = ((mask[0, :, :, 0] > 0.5) * 255).astype(np.uint8)
    mask = cv2.resize(mask, (dim_orig, dim_orig))
    save_image(mask, Path(f'data\\mask'), i)
    seg = cv2.bitwise_and(image, image, mask=mask)
    save_image(seg, Path(f'data\\segmentation'), i)
    return seg


def save_image_classification(
    redes: List[str],
    models: List[Model],
    image_norm,
    image,
    i: int,
    dim_orig: int = 1024,
    labels: List[str] = ['Covid', 'Normal', 'Pneumonia'],
    **params_splits,
) -> None:
    image_norm = image_norm.reshape((1, dim_orig, dim_orig, 1))
    cuts, positions = split_images_n_times(image_norm, **params_splits)
    for model, rede in zip(models, redes):
        print(rede)
        heatmap = prob_grad_cam(
            cuts_images=cuts,
            paths_start_positions=positions,
            model=model,
            dim_orig=dim_orig)
        heatmap = np.uint8(255 * heatmap)
        save_image(heatmap, Path(f'data\\gradCAM\\{rede}'), i)
        grad_cam = plot_gradcam(heatmap, image, dim=dim_orig)
        save_image(grad_cam, Path(f'data\\lungGradCAM\\{rede}'), i)


# %%
def save_image_in_dataset(labels: List[str], image: tfa.types.TensorLike,
                          redes: List[str], model_seg: Model, models: Model,
                          disease: str, survival: str, i: int,
                          **params_splits) -> List[str]:
    classe = class_lung(disease)

    columns = [
        str(Path(f'data\\original') / IMAGE_NAME.format(i)),
        str(Path(f'data\\mask') / IMAGE_NAME.format(i)),
        str(Path(f'data\\segmentation') / IMAGE_NAME.format(i)),
        str(Path(f'data\\gradCAM\\{REDES[0]}') / IMAGE_NAME.format(i)),
        str(Path(f'data\\gradCAM\\{REDES[1]}') / IMAGE_NAME.format(i)),
        str(Path(f'data\\gradCAM\\{REDES[2]}') / IMAGE_NAME.format(i)),
        str(Path(f'data\\gradCAM\\{REDES[3]}') / IMAGE_NAME.format(i)),
        str(Path(f'data\\gradCAM\\{REDES[0]}') / IMAGE_NAME.format(i)),
        str(Path(f'data\\gradCAM\\{REDES[1]}') / IMAGE_NAME.format(i)),
        str(Path(f'data\\gradCAM\\{REDES[2]}') / IMAGE_NAME.format(i)),
        str(Path(f'data\\lungGradCAM\\{REDES[3]}') / IMAGE_NAME.format(i)),
        classe,
        survival
    ]
    save_image(image, Path(f'data\\original'), i)
    image_norm = image / 255
    seg = save_image_segmentation(image=image,
                                  image_norm=image_norm,
                                  i=i,
                                  model_seg=model_seg)
    # save_image_classification(image=seg, \
    #                           image_norm=seg, \
    #                           models=models, \
    #                           labels=labels, \
    #                           redes=redes, \
    #                           i=i,
    #                           **params_splits)
    return columns


def class_lung(disease) -> str:
    if 'COVID-19' in disease:
        return 'Covid-19'
    if 'Pneumonia' in disease:
        return disease
    return 'Normal'


# %%
if __name__ == '__main__':
    rows = []
    lung = Path('ieee_dataset\\lungs')
    ieee_images = ieee_images.reset_index()
    for row in tqdm(range(len(ieee_images))):
        filename = ieee_images['filename'][row]
        if not isinstance(filename, str) or filename is None:
            continue
        image = cv2.imread(str(lung / ieee_images['filename'][row]),
                           cv2.IMREAD_GRAYSCALE)
        transform = compose_original(1024, 1024)
        image = transform(image=image)['image']
        row_dt = save_image_in_dataset(labels=LABELS,
                                       image=image,
                                       redes=REDES,
                                       model_seg=model_seg,
                                       models=models,
                                       disease=ieee_images['finding'][row],
                                       survival=ieee_images['survival'][row],
                                       i=row,
                                       **params_splits)
        rows.append(row_dt)
    df = pd.DataFrame(rows, columns=COLUNM_NAMES)
    df.to_csv('data\\metadata.csv')

    # %%
