from src.images.read_image import read_images
from pathlib import Path
import unittest
import numpy as np
from src.images.process_images import normalize_image
from src.images.process_images import random_pixel
from src.images.process_images import create_recort
from src.images.process_images import create_non_black_cut
from src.images.process_images import split
from src.images.process_images import split_images_n_times
from src.images.process_images import relu

class ProcessImage(unittest.TestCase):

    def test_normalize_image(self):
        image_path = Path('test\images\lena.jpg')
        image = read_images(image_path)
        image = normalize_image(image)
        max_image = np.max(image)
        self.assertTrue(max_image < 1.0)

    def test_random_pixel(self):
        start = (1,2)
        end = (5,5)
        pixel = random_pixel(start, end, 1)
        more_start = pixel[0] >= start[0] and pixel[1] >= start[1]
        less_end = pixel[0] <= end[0] and pixel[1] <= end[1]
        self.assertTrue(more_start and less_end)

    def test_create_recort(self):
        start = (100,100)
        dim = 20
        image_path = Path('test\images\lena.jpg')
        image = read_images(image_path)
        cut = create_recort(image,start,dim)
        cut_shape = [20,20,1]
        self.assertEqual(cut.shape.as_list(),cut_shape)

    def test_create_non_black_cut(self):
        ori_image = np.zeros((100,100))
        ones = np.ones((10,10))
        ori_image[10:20,10:20] = ones
        ori_image[80:90,80:90] = ones
        non_zero_image, _ = create_non_black_cut(ori_image, (0,0), (100,100), 10, 1.0)
        min_non_zero_image = np.min(non_zero_image)
        self.assertEqual(min_non_zero_image, 1.0)

    def test_relu_image(self):
        ori_image = np.random.randn(100,100)
        relu_image = relu(ori_image)
        self.assertEqual(np.min(relu_image), 0.0)

    def test_split_images_n_times(self):
        image = np.random.randn(100,100,1)
        n = 5
        dim = 10
        shape = (n, dim, dim, 1)
        params = {
            'n_split': n, 'dim_split': dim,
            'verbose': True, 'threshold': 0.45
        }
        images, _ = split_images_n_times(image,**params)
        self.assertEqual(images.shape, (n, dim, dim, 1))
