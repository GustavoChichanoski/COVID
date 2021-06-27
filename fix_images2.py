from typing import List, Union
from src.models.segmentacao.segmentacao_model import Unet
from src.dataset.generator_seg import SegmentationDataGenerator
from src.dataset.dataset_seg import SegmentationDataset
from pathlib import Path

def get_all_files(paths: Path) -> Union[Path,List[Path]]:
    if paths.is_dir():
        files = []
        for path in paths.iterdir():
            files.append(get_all_files(path))
        return files
    return paths

DIM = 512
BATCH_SIZE = 2

model = Unet(dim=DIM)
model.compile(loss='log_cosh_dice',rf=10)
model.build()
model.summary()

old_data = Path('D:\\Mestrado\\old_data')
paths = get_all_files(old_data)
peso = 'D:\\Mestrado\\pesos\\pesos.hdf5'

model.load_weights(peso)

model.predict()