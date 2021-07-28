from os import path
from pathlib import Path
from tqdm import tqdm
from src.dataset.dataset import Dataset
from src.images.read_image import read_images

def correct_dataset(
    dataset: Dataset,
    output_folder: Path,
    size: int = 1024,
    rename: bool = True,
    extension: str = 'png'
) -> None:

    for i, path_x in tqdm(enumerate(dataset.x)):
        name = path_x.parts[-1]
        suffix = path_x.suffix
        if suffix == '.png' or suffix == '.jpg' or suffix == '.jpeg':
            image_x = read_images(path_x,dim=size)
        else:
            continue
        if rename:
            name = f'{i:04}.{extension}'