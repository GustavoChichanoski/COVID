import json
import os
import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.models.detection.feature_pyramid import get_backbone
from src.models.detection.label_encoder import LabelEncoder
from src.models.detection.losses import RetinaNetLoss
from src.models.detection.model import RetinaNet
from src.models.segmentation.unet_functional import unet_compile, unet_functional
from src.models.classificacao.funcional_model import classification_model, model_compile
from src.models.severity.dataset import add_new_image


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
        mask_folder=masks_folder,
        dataset_folder=dataset_folder
    )
    if item != None:
        dict_clean[id] = item


dc = pd.DataFrame(dict_clean)
dc = dc.T

dc.to_csv('out.zip', index=False, compression=dict(
    method='zip', archive_name='out.csv'))

model_segmentation = unet_functional(inputs=(256, 256, 1))
unet_compile(model_segmentation)
model_classification = classification_model()
model_compile(model_classification)

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 80
batch_size = 2

learning_rates = [2.5e-6, 625e-6, 1.25e-3, 2.5e-3, 250e-6, 25e-6]
learning_rate_boundaries = [125, 250, 500, 240e3, 360e3]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

# optimizer = SGD(lr=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer='sgd')

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]
