from pathlib import Path
from typing import Dict, List, Tuple
from src.images.read_image import read_images
import pandas as pd

def get_bbox(annotations: pd.DataFrame, image_id: int) -> Tuple[List[int]]:
    lungs = annotations.loc[annotations['image_id'] == image_id]
    right_lung_position = lungs.loc[annotations['category_id'] == 1]
    left_lung_position = lungs.loc[annotations['category_id'] == 2]
    right_lung = right_lung_position['bbox'].values
    left_lung = left_lung_position['bbox'].values
    return (right_lung, left_lung)


def add_new_image(
    id: int,
    df_lung: pd.DataFrame,
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    df_severity: pd.DataFrame,
    lung_folder: Path,
    mask_folder: Path,
    dataset_folder: Path
) -> List[Dict]:

    lung = df_lung.iloc[id]
    lung_filename: Path = Path(lung['filename'])

    filename_lung_images = images['file_name'] == str(lung_filename)
    image_id_serie = images.loc[filename_lung_images]['id']

    right_lung, left_lung = get_lungs_positions(annotations, image_id_serie)

    lung_path, mask_path = lung_mask_path(lung_folder, mask_folder, lung_filename)
    gradCAM_path: Path = dataset_folder / 'gradCAM' / lung_filename

    severity_lung_filename = df_severity['filename'] == str(lung_filename)
    severity_lung = df_severity.loc[severity_lung_filename]
    mask_path = None if not mask_path.exists() else mask_path

    dict_image = {
        'lung': str(lung_path),
        'mask': str(mask_path),
        'grad_cam': str(gradCAM_path.relative_to(Path.cwd())),
        'sex': lung['sex'],
        'age': lung['age'],
        'survival': lung['survival'],
        'intubation': lung['needed_supplemental_O2'],
        'graphics_mean': severity_lung['geographic_mean'].values if not severity_lung.empty else None,
        'opacity_mean': severity_lung['opacity_mean'].values if not severity_lung.empty else None,
        'right_lung_xi': right_lung[0][0],
        'right_lung_yi': right_lung[0][1],
        'right_lung_xf': right_lung[0][2],
        'right_lung_yf': right_lung[0][3],
        'left_lung_xi': left_lung[0][0],
        'left_lung_yi': left_lung[0][1],
        'left_lung_xf': left_lung[0][2],
        'left_lung_yf': left_lung[0][3],
        'id': id
    }
    return dict_image

def lung_mask_path(
    lung_folder: Path,
    mask_folder: Path,
    lung_filename: Path,
) -> Tuple[Path, Path]:
    """Return lung and mask relative path"""
    cwd: Path = Path.cwd()
    lung_suffix: str = lung_filename.suffix
    lung_path: Path = (lung_folder / lung_filename).relative_to(cwd)
    mask_filename_str: str = str(lung_filename).split(lung_suffix)[0]
    mask_filename: Path = Path(mask_filename_str + '_mask' + lung_suffix)
    mask_path: Path = (mask_folder / mask_filename).relative_to(cwd)
    return lung_path, mask_path

def get_lungs_positions(
    annotations: pd.DataFrame,
    image_id_serie
) -> Tuple[List[int], List[int]]:
    if not image_id_serie.empty:
        image_id: int = int(image_id_serie.values)
        right_lung, left_lung = get_bbox(annotations, image_id)
    else:
        none_array = [-1,-1,-1,-1]
        right_lung, left_lung = [none_array], [none_array]
    return right_lung,left_lung
