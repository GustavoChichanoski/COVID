from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import segmentation_models as sm
import tensorflow_addons as tfa

from tensorflow.python.keras import Model

from src.models.losses.log_cosh_dice_loss import LogCoshDiceError

CSV_METADATA = 'metadata.csv'
LUNG = 'lungs'
BACKBONE = 'resnet34'
BATCH_SIZE = 16
EPOCHS = 150
LERNING_RATE = 5e-2
IMG_SIZE = 256
BETA = 0.25
ALPHA = 0.25
GAMMA = 2
SMOOTH = 1
SEED = 42
IMG_SIZE = 256
NCHANNELS = 3
N_CLASSES = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE

UNET_PARAMS = {
    'input_shape': (None, None, 3),
    'encoder_freeze': False,
    'encoder_weights': None,
    'classes': 1,
    'activation': 'sigmoid'
}


def save_image(lung: tfa.types.TensorLike, path: Path, i: int) -> None:
    lung_path = path / LUNG / '{:04d}.png'.format(i)
    cv2.imwrite(str(lung_path), lung)


def compose_train() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(always_apply=False,
                      p=0.4,
                      gamma_limit=(23, 81),
                      eps=1e-07),
        A.RandomBrightnessContrast(always_apply=False,
                                   p=0.4,
                                   brightness_limit=(-0.2, 0.2),
                                   contrast_limit=(-0.2, 0.2),
                                   brightness_by_max=True),
        A.GaussNoise(var_limit=(200, 300), p=0.4),
        A.ElasticTransform(always_apply=False,
                           p=0.4,
                           alpha=3.0,
                           sigma=50.0,
                           alpha_affine=50.0,
                           interpolation=0,
                           border_mode=0,
                           value=(0, 0, 0),
                           mask_value=None,
                           approximate=False),
        A.GridDistortion(always_apply=False,
                         p=0.4,
                         num_steps=5,
                         distort_limit=(-0.3, 0.3),
                         interpolation=0,
                         border_mode=0,
                         value=(0, 0, 0),
                         mask_value=None)
    ])


def create_dataset(df: pd.DataFrame, path: Path, new_path: Path,
                   column_filename: str, df_ieee: pd.DataFrame) -> None:
    transform_ori = compose_original()
    dataset = []
    i = 0
    for index, row in tqdm(df.iterrows()):
        filename = path / row[column_filename]
        survival = df_ieee.loc[df_ieee['filename'] == str(
            filename)]['filename']
        if filename.exists():
            image = cv2.imread(str(filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            i, dataset = save_image_folder_keras(new_path=new_path,
                                                 transform_ori=transform_ori,
                                                 dataset=dataset,
                                                 i=i,
                                                 row=row,
                                                 image=image,
                                                 filename=filename,
                                                 survival=survival)
    df_class = pd.DataFrame(dataset,
                            columns=[
                                'lung', 'disease', 'filename', 'px_start_x',
                                'px_start_y', 'px_end_x', 'px_end_y',
                                'survival'
                            ])
    df_class.to_csv(new_path / CSV_METADATA)


def compose_original(height: int = 1024, width=1024) -> A.Compose:
    return A.Compose([
        A.Resize(height=height, width=width, always_apply=True),
        A.Equalize(always_apply=True),
        A.InvertImg(always_apply=True)
    ])


def save_image_folder(
        new_path: Path, transform_ori: A.Compose, dataset: pd.DataFrame,
        i: int, row: pd.Series,
        image: tfa.types.TensorLike) -> Tuple[int, List[List[str]]]:
    image = transform_ori(image=image)['image']
    pred = model.predict(image)
    save_image(pred[0, :, :, 0], new_path, i)
    dataset.append([
        str(new_path / row['filename']), row['survival'],
        '{:04d}.png'.format(i)
    ])
    i += 1
    return i, dataset


def calc_severity(survival: str, disease: str, intubated: str) -> int:
    score = 0
    if disease == 'Normal':
        return 0
    if survival == 'N':
        score += 1
    if intubated == 'Y':
        score += 1
    return score


def save_image_folder_keras(new_path: Path,
                            transform_ori: A.Compose,
                            dataset: pd.DataFrame,
                            i: int,
                            row: pd.Series,
                            image: tfa.types.TensorLike,
                            filename: str,
                            survival: str = '') -> Tuple[int, List[List[str]]]:
    image = transform_ori(image=image)['image']
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    resized = resized.reshape((1, 256, 256, 1))
    pred = model.predict(resized / 255)
    mask = cv2.resize(pred[0, :, :, 0], (1024, 1024),
                      interpolation=cv2.INTER_AREA)
    mask = mask.reshape((1024, 1024))
    mask = (mask > 0.5).astype(np.uint8)
    if len(image.shape) > 2:
        y_nonzero, x_nonzero = np.nonzero(mask[:, :, 0])
    else:
        y_nonzero, x_nonzero = np.nonzero(mask[:, :])
    px_start, px_end = (np.min(y_nonzero), np.min(x_nonzero)), \
                        (np.max(y_nonzero), np.max(x_nonzero))
    seg = cv2.bitwise_and(image, image, mask=mask)
    save_image(seg, new_path, i)
    label = ""
    if row['label'] == 'Normal':
        label = 'Normal'
    else:
        label += row['virus_category'] if not pd.isna(
            row['virus_category']) else ""
        label += row['virus_category2'] if pd.isna(
            row['virus_category']) else ""
    survival = '' if survival is None else survival
    dataset.append([
        str(new_path / row['filename']), label, row['type'], filename,
        px_start[0], px_start[1], px_end[0], px_end[1], survival
    ])
    i += 1
    return i, dataset


def remove_dump_colunms(csv_file: str, dataset_ieee: Path) -> pd.DataFrame:
    df_ieee = pd.read_csv(dataset_ieee / csv_file)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['finding'] == 'todo'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['view'] == 'L'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['view'] == 'Axial'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['finding'] == 'todo'].index)
    return df_ieee


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
    # dice_loss = sm.losses.DiceLoss()
    # focal_loss = sm.losses.BinaryFocalLoss()
    # total_loss = focal_loss + dice_loss
    log_dice = LogCoshDiceError(regularization_factor=100)
    model.compile(opt,
                  loss=log_dice,
                  metrics=[sm.metrics.IOUScore(),
                           sm.metrics.FScore()])


if __name__ == '__main__':

    dataset_ieee = Path('ieee_dataset')
    lung_folder = dataset_ieee / LUNG

    dataset_path = Path('dataset')
    dataset_keras = dataset_path / 'class_keras'
    df_keras = pd.read_csv(dataset_keras / CSV_METADATA)

    new_path = dataset_path / 'class_keras_pre_process'

    df_ieee = remove_dump_colunms(CSV_METADATA, Path('ieee_dataset'))
    model = create_segmentation_model(BACKBONE, UNET_PARAMS, IMG_SIZE)
    segmentation_model_compile(model)
    model.load_weights('.\\kaggle\\model.hdf5')

    create_dataset(df=df_keras,
                   path=dataset_keras / LUNG,
                   new_path=new_path,
                   column_filename='filename',
                   df_ieee=df_ieee)
