"""
    Bibliotecas referente aos plots do projeto
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2 as cv
import pandas as pd
import tensorflow_addons as tfa

from pathlib import Path

from tensorflow.python.keras import Model

from src.data.classification.cla_generator import ClassificationDatasetGenerator

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
