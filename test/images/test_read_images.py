from src.images.read_image import read_images
from pathlib import Path
import unittest

class ReadImage(unittest.TestCase):

    def test_read_images_exist_image(self):
        cwd = Path.cwd()
        path_image_exists = cwd / "test" / "images" / "lena.jpg"
        
        image = read_images(path_image_exists, dim=225)
        self.assertEqual(image.shape.as_list(), [225,225,1])
    
    def test_read_images_resize_image(self):
        cwd = Path.cwd()
        path_image_exists = cwd / "test\images\lena.jpg"
        
        image = read_images(path_image_exists, dim=1024)
        self.assertEqual(image.shape.as_list(), [1024,1024,1])

    def test_read_images_color_image(self):
        cwd = Path.cwd()
        path_image_exists = cwd / "test\images\lena.jpg"
        
        image = read_images(path_image_exists, color=True, dim=1024)
        self.assertEqual(image.shape, (1024,1024,3))

    def test_read_images_list_image(self):
        cwd = Path.cwd()
        path_image_exists = cwd / "test\images\lena.jpg"
        n = 5

        list_images = [path_image_exists for i in range(0, n)]

        image = read_images(list_images, 0, n, color=True, dim=1024)
        self.assertEqual(image.shape, (n, 1024, 1024, 3))

    def test_read_images_list_images_without_end(self):
        cwd = Path.cwd()
        path_image_exists = cwd / "test\images\lena.jpg"
        
        list_images = [path_image_exists for _ in range(0, 10)]

        image = read_images(list_images, 0, color=True, dim=1024)
        self.assertEqual(image.shape, (10, 1024, 1024, 3))

    def test_read_images_list_images_random(self):
        cwd = Path.cwd()
        path_image_exists = cwd / "test\images\lena.jpg"

        list_images = [path_image_exists for _ in range(0, 10)]

        image = read_images(list_images, id_start=[0, 1], color=True, dim=1024)
        self.assertEqual(image.shape, (2, 1024, 1024, 3))

if __name__ == '__main__':
    unittest.main()