from pathlib import Path
from labelbox import Client, Project
import requests
from getpass import getpass
from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, Any
from labelbox.schema.ontology import Tool, OntologyBuilder
import os
import cv2

import matplotlib.pyplot as plt

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3Iwamw4djYzYzNiMHlkamJ1cHJjc2NjIiwib3JnYW5pemF0aW9uSWQiOiJja3Iwamw4cjMzYzNhMHlkajgxd3djZG83IiwiYXBpS2V5SWQiOiJja3IxMGd1Mjl1NGRiMHllNDhjbGczZmNxIiwic2VjcmV0IjoiODRmZGM4ODU5YTNjN2EyMzEzMzBiM2QzMjFkZGE2MDQiLCJpYXQiOjE2MjYxMTc2NTYsImV4cCI6MjI1NzI2OTY1Nn0.Wce-GvpS2scj1akCsNCq-lg8Y0MemPQwpz5gA-hdWKA"

PROJECT_KEY = "ckr0lie4zapp00yar0311afu0"

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
j = 704

path = Path("D:\\Mestrado\\new_data\\train")

for i in range(length_exports):
    content = BytesIO(requests.get(exports[i]["Labeled Data"]).content)
    image = np.array(Image.open(content))
    if len(image.shape) > 2:
        image = np.array(Image.open(content))[:,:,0]

    lung = cv2.resize(image,(1024,1024))
    mask = np.zeros((1024,1024)).astype(np.uint8)

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