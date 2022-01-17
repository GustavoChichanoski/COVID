from __future__ import annotations
import json
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path
from src.images.read_image import read_images


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
    mask_folder: Path
) -> List[Dict]:

    lung = df_lung.iloc[id]
    lung_filename: Path = Path(lung['filename'])

    image_id_serie = images.loc[images['file_name'] == str(lung_filename)]['id']
    if not image_id_serie.empty:
        image_id = int(image_id_serie.values)
        right_lung, left_lung = get_bbox(annotations, image_id)
    else:
        right_lung, left_lung = [[None,None,None,None]], [[None,None,None,None]]

    lung_path: Path = (lung_folder / lung_filename).relative_to(Path.cwd())
    mask_filename: Path = Path(str(lung_filename).split(lung_filename.suffix)[0] + '_mask' + lung_filename.suffix)
    mask_path: Path = (mask_folder / mask_filename).relative_to(Path.cwd())
    gradCAM_path: Path = dataset_folder / 'gradCAM' / lung_filename

    severity_lung = df_severity.loc[df_severity['filename'] == str(lung_filename)]
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


cwd = Path.cwd()
ieee_folder = cwd / "ieee_dataset"
severity_folder = ieee_folder / "annotations"

csv_severity = severity_folder / "covid-severity-scores.csv"
img_severity_folder = severity_folder / "lungVAE-masks"

df_severity = pd.read_csv(csv_severity)
df_metadata = pd.read_csv(ieee_folder / "metadata.csv")

dataset_folder = cwd / 'ieee_dataset'
lung_folder = dataset_folder / 'lungs'
masks_folder = dataset_folder / 'annotations' / 'lungVAE-masks'

json_mask_file = severity_folder / "imageannotation_ai_lung_bounding_boxes.json"

with open(json_mask_file, 'r') as read_file:
    data = json.load(read_file)

images = pd.DataFrame(data['images'])
annotations = pd.DataFrame(data['annotations'])

dict_clean = {}

for id in range(len(df_metadata.index)):
    item = add_new_image(
        id=id,
        df_lung=df_metadata,
        images=images,
        annotations=annotations,
        df_severity=df_severity,
        lung_folder=lung_folder,
        mask_folder=masks_folder
    )
    if item != None:
        dict_clean[id] = item


dc = pd.DataFrame(dict_clean)
dc = dc.T

dc.to_csv('out.zip', index=False, compression=dict(method='zip', archive_name='out.csv'))