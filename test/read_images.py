import unittest
from src.images.read_image import read_images as ri

class TestStringMethods(unittest.TestCase):

    def test_single_image(self):
        path = 'data/train/Covid/0000.png'
        image = ri(path)

if __name__ == '__main__':
    unittest.main()