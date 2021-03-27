"""
    Bibliotecas referente aos plots do projeto
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2 as cv
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array

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

def plot_gradcam(heatmap,image, save: bool = False):
    """ Plota o gradCam probabilstico recebendo como parametro o
        mapa de calor e a imagem original. Ambos de mesmo tamanho.

        Args:
        -----
            heatmap (np.array): Mapa de calor
            image (np.array): Imagem original
    """
    heatmap8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_color = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_color[heatmap8]

    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((1024, 1024))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_image = jet_heatmap * 0.4 + image
    superimposed_image = array_to_img(superimposed_image)

    plt.imshow(superimposed_image)
    plt.show()