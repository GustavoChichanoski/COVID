import unittest
import numpy as np
from src.images.data_augmentation import random_rotate_image
from src.images.data_augmentation import flip_vertical_image
from src.images.data_augmentation import flip_horizontal_image

class TestDataAugmentation(unittest.TestCase):

    def test_rotate_image(self):
        image = np.ones((100,100))
        zeros = np.zeros((20,20))
        image[20:40,20:40] = zeros
        image = random_rotate_image(image,angle=5)
        valid_image = image[0,0] == 0
        self.assertTrue(valid_image)

    def test_flip_vertical_image(self):
        image = np.ones((100,100,1))
        zeros = np.zeros((20,20,1))
        image[20:40,20:40] = zeros
        flip_image = flip_vertical_image(image)
        valid_image = flip_image[70,30,0] == 0
        self.assertTrue(valid_image)

    def test_flip_horizontal_image(self):
        image = np.ones((100,100,1))
        zeros = np.zeros((20,20,1))
        image[20:40,20:40] = zeros
        image = flip_horizontal_image(image)
        self.assertTrue(image[30,70,0] == 0)