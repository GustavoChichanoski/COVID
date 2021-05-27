"""
    Bibliotecas referente aos plots do projeto
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2 as cv
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras import Model

def plot_model(path: Path, model: Model) -> str:
    for layers in model.layers:
        for layer in layers.layers:
            config = layer.get_config()
    return "None"

def plot_images(images, cmap:str ='gray'):
    """Plotas as imagens passafas em images
        Args:
            images (list or np.array): imagens a serem plotadas
    """
    if isinstance(images, list):
        for img in images:
            plot_images(img, cmap)
    else:
        plt.imshow(images,cmap)
        plt.show()

def plot_gradcam(
    heatmap,
    image,
    grad: bool = True,
    name: str = None,
    dim: int = 1024,
    alpha = 0.4
) -> str:
    """ Plota o gradCam probabilstico recebendo como parametro o
        mapa de calor e a imagem original. Ambos de mesmo tamanho.

        Args:
        -----
            heatmap (np.array): Mapa de calor
            image (np.array): Imagem original
    """
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_color = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_color[heatmap]

    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((dim, dim))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_image = jet_heatmap * alpha + image
    superimposed_image = array_to_img(superimposed_image)

    fig = plt.figure()
    plt.imshow(superimposed_image)
    plt.show()
    # Salvar imagem
    path = ''
    if name is not None:
        path = '{}.png'.format(name)
        plt.savefig(path,dpi=fig.dpi)
    if grad: plt.show()
    return path
