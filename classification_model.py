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
def create_segmentation_model(backbone: str,
                              dim_input: int,
                              **unet_params) -> Model:
    """gera o modelo de segmentação do keras

    Args:
        backbone (str): nome do backbone utilizado
        dim_input (int): dimensão de entrada

    Returns:
        Model: modelo de segmentação
    """
    input_layer = tf.keras.Input((dim_input, dim_input, 1))
    layer = tf.keras.layers.Conv2D(filters=3,
                                   kernel_size=3,
                                   padding='same',
                                   activation='relu')(input_layer)
    layer = sm.Unet(backbone, **unet_params)(layer)
    model = tf.keras.Model(inputs=input_layer, outputs=layer)
    return model


def segmentation_model_compile(model: Model) -> None:
    """ Modelo de segmentação compilado

    Args:
        model (Model): modelo do keras
    """
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    log_dice = LogCoshDiceError(regularization_factor=100)
    model.compile(opt,
                  loss=log_dice,
                  metrics=[sm.metrics.IOUScore(),
                           sm.metrics.FScore()])


def remove_dump_colunms(csv_file: str, dataset_ieee: Path) -> pd.DataFrame:
    """remove as colunas erradas do csv

    Args:
        csv_file (str): arquivo csv para ser lido
        dataset_ieee (Path): diretorio do csv

    Returns:
        pd.DataFrame: retorna o dataset removido
    """
    df_ieee = pd.read_csv(dataset_ieee / csv_file)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['finding'] == 'todo'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['view'] == 'L'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['view'] == 'Axial'].index)
    df_ieee = df_ieee.drop(df_ieee[df_ieee['finding'] == 'todo'].index)
    return df_ieee


def save_image(lung: tfa.types.TensorLike,
               path: Path) -> None:
    """Salva uma imagem no diretorio path com o nome de `image_name_0i.png`

    Args:
        lung (tfa.types.TensorLike): _description_
        path (Path): _description_
        i (int): _description_
        image_name (str, optional): _description_. Defaults to '{:04}.png'.
    """
    cv2.imwrite(str(path), lung)

# %%


def compose_original(height: int = 1024, width: int = 1024) -> A.Compose:
    """Retorna a imagem redimensionada, equalizada e com as cores invertidas.

    Args:
        height (int, optional): altura da imagem de saída. Defaults to 1024.
        width (int, optional): comprimento da imagem de saída. Defaults to 1024.

    Returns:
        A.Compose: gerador de composição.
    """
    return A.Compose([
        A.Resize(height=height, width=width, always_apply=True),
        A.Equalize(always_apply=True),
        A.InvertImg(always_apply=True)
    ])


def save_image_segmentation(model_seg: Model,
                            image_norm: tfa.types.TensorLike,
                            image: tfa.types.TensorLike,
                            columns: List[str],
                            dim_spit: int = 256,
                            dim_orig: int = 1024) -> tfa.types.TensorLike:
    """Salva a mascara e a imagem segmentada do pulmão.

    Args:
        i (int): número de imagens já utilizadas
        model_seg (Model): modelo de segmentação
        image_norm (tfa.types.TensorLike): imagem já normalizada
        image (tfa.types.TensorLike): imagem não normalizada
        dim_spit (int, optional): dimensão final. Defaults to 256.
        dim_orig (int, optional): dimensão de entrada. Defaults to 1024.

    Returns:
        tfa.types.TensorLike: imagem segmentada
    """
    # dimensão do corte
    split_shape = (dim_spit, dim_spit)
    # redimensionamento da imagem do pulmão
    image_model_seg = cv2.resize(image_norm, split_shape)
    # redimensionamento para passar pelo modelo de segmentação
    image_model_seg = image_model_seg.reshape((1, dim_spit, dim_spit, 1))
    # geração da mascara do pulmão
    mask = model_seg.predict(image_model_seg)
    # convertendo a mascara de float32 para uint8 com os valores de 0 e 255
    mask = ((mask[0, :, :, 0] > 0.5) * 255).astype(np.uint8)
    # redimensionando uma imagem
    mask = cv2.resize(mask, (dim_orig, dim_orig))
    save_image(mask, Path.cwd() / columns[1])
    # mascara da imagem
    seg = cv2.bitwise_and(image, image, mask=mask)
    save_image(seg, Path.cwd() / columns[2])
    return seg


def save_image_classification(redes: List[str],
                              models: List[Model],
                              image_norm: tfa.types.TensorLike,
                              image: tfa.types.TensorLike,
                              columns: List[str],
                              dim_orig: int = 1024,
                              labels: List[str] = LABELS,
                              **params_splits) -> None:
    """ Gera os arquivos de mapa de calor e o Grad Cam Probabilistico do pulmão.

    Args:
        redes (List[str]): nomes das redes de classificação
        models (List[Model]): modelos das redes de classificação
        image_norm (tfa.types.TensorLike): imagem normalizada
        image (tfa.types.TensorLike): imagem original
        columns (List[str]): paths dos arquivos a serem criados
        dim_orig (int, optional): dimensão da imagem original. Defaults to 1024.
    """
    # dimensão do corte
    split_shape = (1, dim_orig, dim_orig, 1)
    # redimensionamento da imagem gerada
    image_norm = image_norm.reshape(split_shape)
    # criação dos recortes da imagem
    cuts, positions = split_images_n_times(image_norm, **params_splits)
    i = 4
    # geração fos heatmap e criação do grad cam probabilistico para cada modelo
    for model, rede in zip(models, redes):
        # mapa de calor para os cortes
        heatmap = prob_grad_cam(
            cuts_images=cuts,
            paths_start_positions=positions,
            model=model,
            dim_orig=dim_orig)
        # mapa de calor da imagem
        heatmap = np.uint8(255 * heatmap)
        # cria um arquivo com a imagem do heatmap
        save_image(heatmap, columns[i])
        i += 1
        # cria o Grad Cam Probabilistico
        grad_cam = plot_gradcam(heatmap, image, dim=dim_orig)
        # cria um arquivo com do grad cam probabilistico
        save_image(grad_cam, columns[i])
        i += 1

# %%
def save_image_in_dataset(labels: List[str],
                          image: tfa.types.TensorLike,
                          redes: List[str],
                          model_seg: Model,
                          models: Model,
                          disease: str,
                          survival: str,
                          i: int,
                          **params_splits) -> List[str]:
    # define a classe da imagem do pulmão
    classe = class_lung(disease)
    # cria a lista das colunas a serem armazenadas no dataset
    columns = lista_imagem_criada(survival=survival,
                                  i=i,
                                  classe=classe,
                                  redes=redes)
    # salva a imagem original no caminho
    save_image(lung=image, path=Path.cwd() / columns[0])
    # normalização da imagem
    image_norm = image / 255
    # gravação das imagens nas pastas
    seg = save_image_segmentation(image=image,
                                  image_norm=image_norm,
                                  columns=columns,
                                  model_seg=model_seg)
    # gravação das imagens de classicação nas pastas
    # save_image_classification(image=seg,
    #                           image_norm=seg,
    #                           models=models,
    #                           labels=labels,
    #                           redes=redes,
    #                           columns=columns,
    #                           **params_splits)
    return columns

def lista_imagem_criada(survival:str,
                        i: int,
                        classe: str,
                        redes: List[str],
                        image_name: str = None,) -> List[str]:
    if image_name is None:
        image_name = IMAGE_NAME
    # cria as strings de nome
    data = Path('data')
    name_file = image_name.format(i)
    # popula a lista de nomes da coluna
    columns = [str(data / 'original' / name_file)]
    columns.append(str(data / 'mask' / name_file))
    columns.append(str(data / 'segmentation' / name_file))
    for rede in redes:
        columns.append(str(data / 'gradCAM' / rede / name_file))
        columns.append(str(data / 'lungGradCAM' / rede / name_file))
    columns.append(classe)
    columns.append(survival)
    return columns


def class_lung(disease: str) -> str:
    if 'COVID-19' in disease:
        return 'Covid-19'
    if 'Pneumonia' in disease or 'Lung_Opacity' in disease:
        return disease
    return 'Normal'

def read_dataset(dataset: pd.DataFrame,
                 data_path: Path,
                 redes: List[str],
                 model_seg: Model,
                 models: List[Model],
                 transform: A.Compose,
                 rows: List[List[str]],
                 i: int = 0,
                 labels: List[str] = LABELS,
                 **params_splits) -> List[List[str]]:
    for row in tqdm(range(len(dataset))):
        # carrega o caminho do arquivo
        filename = dataset['filename'][row]
        if not isinstance(filename, str) or filename is None:
            continue
        # carrega a imagem do pulmão
        imagem = cv2.imread(str(data_path / filename), cv2.IMREAD_GRAYSCALE)
        # realização a transformação na imagem
        imagem = transform(image=imagem)['image']
        row_dt = save_image_in_dataset(labels=labels,
                                       image=imagem,
                                       redes=redes,
                                       model_seg=model_seg,
                                       models=models,
                                       disease=dataset['class'][row],
                                       survival=dataset['survival'][row],
                                       i=i,
                                       **params_splits)
        rows.append(row_dt)
        i += 1
    return rows
# %%
def read_ieee_dataframe(redes: List[str],
                        labels: List[str],
                        ieee_images: pd.DataFrame,
                        model_seg: Model,
                        models: List[Model],
                        transform: A.Compose,
                        rows: List[List[str]],
                        lung: Path,
                        i: int = 0,
                        **params_splits):
    for row in tqdm(range(len(ieee_images))):
        # carrega o caminho do arquivo
        filename = ieee_images['filename'][row]
        if not isinstance(filename, str) or filename is None:
            continue
        # carrega a imagem do pulmão
        image = cv2.imread(str(lung / filename),
                           cv2.IMREAD_GRAYSCALE)
        # realiza a transformação na imagem
        image = transform(image=image)['image']

        row_dt = save_image_in_dataset(labels=labels,
                                       image=image,
                                       redes=redes,
                                       model_seg=model_seg,
                                       models=models,
                                       disease=ieee_images['finding'][row],
                                       survival=ieee_images['survival'][row],
                                       i=i,
                                       **params_splits)
        i += 1
        rows.append(row_dt)

    return rows


if __name__ == '__main__':

    # leitura do arquivos
    ieee = Path('ieee_dataset')
    ieee_images = pd.read_csv(ieee / 'metadata.csv')
    ieee_annotations = pd.read_csv(ieee / 'annotations'
                                   / 'covid-severity-scores.csv')
    ieee_images = remove_dump_colunms('metadata.csv', ieee)

    # %%
    dataset = Path('data')
    orig = dataset / 'original'
    mask = dataset / 'mask'
    seg = dataset / 'segmentation'
    grad = dataset / 'gradCAM'
    lungGradCAM = dataset / 'lungGradCAM'
    # %% segmentation model
    model_seg = create_segmentation_model(BACKBONE, IMG_SIZE, **UNET_PARAMS)
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
    rows = []
    lung = Path('ieee_dataset\\lungs')
    ieee_images = ieee_images.reset_index()
    transform = compose_original(height=DIM_ORIGINAL, width=DIM_ORIGINAL)
    i = 0
    read_ieee_dataframe(redes=REDES,
                        labels=LABELS,
                        ieee_images=ieee_images,
                        model_seg=model_seg,
                        models=models,
                        rows=rows,
                        lung=lung,
                        transform=transform,
                        i = i,
                        **params_splits,)
    path_data = Path.cwd()
    dataset = pd.read_csv('.\dataset\COVID-19_Radiography_Dataset\metadata.csv')
    read_dataset(redes=REDES,
                 labels=LABELS,
                 dataset=dataset,
                 model_seg=model_seg,
                 models=models,
                 rows=rows,
                 data_path=path_data,
                 transform=transform,
                 i = i,
                 **params_splits)
    df = pd.DataFrame(rows, columns=COLUNM_NAMES)
    df.to_csv('data\\metadata.csv')

    # %%
