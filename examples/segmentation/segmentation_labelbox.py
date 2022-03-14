from tqdm import tqdm
from pathlib import Path
from labelbox import Client, Project
from getpass import getpass
from PIL import Image
from pathlib import Path
from src.data.segmentation.dataset_seg import SegmentationDataset
from src.images.read_image import read_images
from io import BytesIO
from typing import Dict, Any, Tuple
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

PROJECT_KEY = "ckrj8mgsri6ty0y7r7jke4kcy"
DIM = 1024

def get_position_box(tool: Dict[str, Any]) -> np.ndarray:
    start = (tool['bbox']['left'], tool['bbox']["top"])
    end = (tool["bbox"]["left"] + tool["bbox"]["width"], tool["bbox"]["top"] + tool["bbox"]["height"])
    return (start,end)

def visualize_bbox(image: np.ndarray, start: Tuple[int,int], end: Tuple[int,int]) -> np.ndarray:
    """
        Draws a bounding box on an image.

        Args:
            image (np.ndarray): image to draw a bounding box onto
            tool (Dict[str, Any]): Dict resonse from the export
        Returns:
            image with a bounding box drawn on it.
    """
    return cv2.rectangle(image, start, end, (255,0,0), -1)

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

path_root = Path('D:\\Mestrado')
train_path = Path('block_segmentation\\train')
val_path = Path('block_segmentation\\val')
test_path = Path('block_segmentation\\test')
metadata = path_root / 'block_segmentation\\metadata.csv'

dataset = SegmentationDataset(
    path_lung = data_path / 'CXR_png',
    path_mask = data_path / 'masks'
)

tamanho_dataset = len(dataset.x)

j = 0

start_left = (0,0)
start_right = (0,0)

data = np.array([])

for i in tqdm(range(length_exports)):

    content = BytesIO(requests.get(exports[i]["Labeled Data"]).content)

    image = np.array(Image.open(content))

    if len(image.shape) > 2:
        image = np.array(Image.open(content))[:,:,0]

    mask = np.zeros(image.shape).astype(np.uint8)

    end_left = (0,0)
    end_right = (0,0)

    dataset_length = len(exports)

    test = int(0.2 * dataset_length)
    train = int(0.8 * (dataset_length - test))
    val = dataset_length - test - train

    for tool in exports[i]["Label"]["objects"]:
        if "bbox" in tool:
            start, end = get_position_box(tool)
            mask = visualize_bbox(mask, start, end)
            if tool['title'] == 'lung_left':
                start_left = start
                end_left = end
            if tool['title'] == 'lung_right':
                start_right = start
                end_right = end
        # if "instanceURI" in tool and (tool["title"] == "left" or tool["title"] == "right") :
        #     mask = visualize_mask(mask, tool)
        # if "polygon" in tool:
        #     image = visualize_polygon(image, tool)

    if i < val:
        new_data_path = val_path
    elif i < val + test:
        new_data_path = test_path
    else:
        new_data_path = train_path

    new_data_path_lung = new_data_path / 'lungs'
    new_data_path_mask = new_data_path / 'masks'

    lung_path = new_data_path_lung / f'{j:04d}.png'
    mask_path = new_data_path_mask / f'{j:04d}.png'

    data = np.append(
        data,
        [
            lung_path,
            mask_path,
            start_left[0],
            start_left[1],
            end_left[0],
            end_left[1],
            start_right[0],
            start_right[1],
            end_right[0],
            end_right[1]
        ]
    )

    lung = cv2.resize(image,(DIM,DIM))
    mask = cv2.resize(mask,(DIM,DIM))

    cv2.imwrite(str(path_root / lung_path), lung)
    cv2.imwrite(str(path_root / mask_path), mask)

    j += 1

number_files = int(data.size / 10)
data = np.reshape(data,(number_files,10))

df = pd.DataFrame(
    data,
    columns=[
        'lung',
        'mask',
        'start_left_x',
        'start_left_y',
        'end_left_x',
        'end_left_y',
        'start_right_x',
        'start_right_y',
        'end_right_x',
        'end_right_y'
    ]
)

df.to_csv(metadata)
