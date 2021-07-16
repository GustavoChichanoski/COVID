from pathlib import Path
from labelbox import Client, Project
from getpass import getpass
from PIL import Image
from pathlib import Path
from src.dataset.dataset_seg import SegmentationDataset
from src.images.read_image import read_images
from io import BytesIO
from typing import Dict, Any
from labelbox.schema.ontology import Tool, OntologyBuilder
import pandas as pd
import numpy as np
import os
import cv2
import cv2 as cv
import requests
import numpy as np
import matplotlib.pyplot as plt

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3Iwamw4djYzYzNiMHlkamJ1cHJjc2NjIiwib3JnYW5pemF0aW9uSWQiOiJja3Iwamw4cjMzYzNhMHlkajgxd3djZG83IiwiYXBpS2V5SWQiOiJja3IxMGd1Mjl1NGRiMHllNDhjbGczZmNxIiwic2VjcmV0IjoiODRmZGM4ODU5YTNjN2EyMzEzMzBiM2QzMjFkZGE2MDQiLCJpYXQiOjE2MjYxMTc2NTYsImV4cCI6MjI1NzI2OTY1Nn0.Wce-GvpS2scj1akCsNCq-lg8Y0MemPQwpz5gA-hdWKA"

PROJECT_KEY = "ckr0lie4zapp00yar0311afu0"
DIM = 1024

def visualize_bbox(image: np.ndarray, tool: Dict[str, Any]) -> np.ndarray:
    """
        Draws a bounding box on an image.

        Args:
            image (np.ndarray): image to draw a bounding box onto
            tool (Dict[str, Any]): Dict resonse from the export
        Returns:
            image with a bounding box drawn on it.
    """
    start = (tool['bbox']['left'], tool['bbox']["top"])
    end = (tool["bbox"]["left"] + tool["bbox"]["width"],
           tool["bbox"]["top"] + tool["bbox"]["height"])
    return cv2.rectangle(image, start, end, (255,0,0), 5)

def visualize_mask(image: np.ndarray, tool: Dict[str, Any], alpha: float = 0.5) -> np.ndarray:
    """
        Overlay a mask onto a image.

        Args:
            image (np.ndarray): image to overlay mask onto.
            tool (Dict[str, Any]): Dict response from the export.
            alpha (float): How much to weight the when adding to the image.
        Return:
            image with a point drawn on it.
    """
    mask = np.array(Image.open(BytesIO(requests.get(tool["instanceURI"]).content)))[:,:,0]
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask,(1024,1024))
    image = cv2.bitwise_or(image, mask)
    return image

def visualize_point(image: np.ndarray, tool: Dict[str, Any]) -> np.ndarray:
    """ Draw a point on an image.

        Args:
            image (np.ndarray): image to draw a point onto
            tool (Dict[str, Any]): Dict response from the export

        Returns:
            np.ndarray: image with a point draw onto it.
    """
    return cv2.circle(
        image,
        (tool["point"]["x"],tool["point"]["y"]),
        radius=10,
        color=(0,0,255),
        thickness=5
    )

def visualize_polygon(image: np.ndarray, tool: Dict[str, Any]) -> np.ndarray:
    """
        Draws a polygon on an image.

        Args:
            image (np.ndarray): image to drwa a polygon onto
            tool (Dict[str, Any]): Dict response from the export.
        Returns:
            iamge with a polygon drawn on it.
    """
    poly = [[pt["x"], pt["y"]] for pt in tool["polygon"]]
    poly = np.array(poly)
    return cv2.polylines(image, [poly], True, (0,255,0), thickness=5)

client = Client(API_KEY)
project = client.get_project(PROJECT_KEY)
export_url = project.export_labels()
print(export_url)

exports = requests.get(export_url).json()
length_exports = len(exports)

path = Path("D:\\Mestrado\\new_data\\train")

data_path = Path('D:\\Mestrado\\data\\Lung Segmentation')
new_data_path = Path('D:\\Mestrado\\data_segmentation\\train')

dataset = SegmentationDataset(
    path_lung = data_path / 'CXR_png',
    path_mask = data_path / 'masks'
)

new_data_path_lung = new_data_path / 'lungs'
new_data_path_mask = new_data_path / 'masks'

tamanho_dataset = len(dataset.x)

j = 0

for (path_lung, path_mask) in zip(dataset.x, dataset.y):
    print(f"Imagem {j}")
    lung = read_images(path_lung, dim=DIM)
    mask = read_images(path_mask, dim=DIM)

    lung = lung.numpy().astype(np.uint8)
    mask = mask.numpy().astype(np.uint8)

    cv2.imwrite(f'{new_data_path_lung}/{j:04}.png',lung)
    cv2.imwrite(f'{new_data_path_mask}/{j:04}.png',mask)
    j += 1

for i in range(length_exports):

    print(f"Imagem {j}")

    content = BytesIO(requests.get(exports[i]["Labeled Data"]).content)
    image = np.array(Image.open(content))
    if len(image.shape) > 2:
        image = np.array(Image.open(content))[:,:,0]

    lung = cv2.resize(image,(DIM,DIM))
    mask = np.zeros((DIM,DIM)).astype(np.uint8)

    for tool in exports[i]["Label"]["objects"]:
        # if "bbox" in tool:
        #     image = visualize_bbox(image, tool)
        if "instanceURI" in tool and (tool["title"] == "left" or tool["title"] == "right") :
            mask = visualize_mask(mask, tool)
        # if "polygon" in tool:
        #     image = visualize_polygon(image, tool)

    lung_path = path / 'lungs' / f'{j:04d}.png'
    mask_path = path / 'masks' / f'{j:04d}.png'

    cv2.imwrite(str(lung_path), lung)
    cv2.imwrite(str(mask_path), mask)

    j += 1