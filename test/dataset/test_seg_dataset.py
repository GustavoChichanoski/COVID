import unittest
from src.data.segmentation.dataset_seg import SegmentationDataset
from pathlib import Path
import numpy as np


class TestDatasetSeg(unittest.TestCase):

    def test_create_dataset(self):
        data_path = Path('D:\\Mestrado\\datasets\\Lung Segmentation')
        test_ds = SegmentationDataset(
            path_lung=data_path / 'CXR_png',
            path_mask=data_path / 'masks'
        )
        equal_len = len(test_ds.x) == len(test_ds.y)
        self.assertTrue(len(test_ds.x) > 0 and equal_len)

    def test_partition(self):
        data_path = Path('D:\\Mestrado\\datasets\\Lung Segmentation')
        test_ds = SegmentationDataset(
            path_lung=data_path / 'CXR_png',
            path_mask=data_path / 'masks'
        )
        train, val = test_ds.partition(val_size=0.2)
        len_x_train = len(train[0])
        len_x_val = len(val[0])
        self.assertEqual(len_x_train + len_x_val, len(test_ds.x))
        self.assertEqual(int(np.ceil(len(test_ds.x) * 0.2)), len_x_val)
