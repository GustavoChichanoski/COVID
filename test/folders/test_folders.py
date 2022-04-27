from src.output_result.folders import *
from pathlib import Path
import unittest

class Folders(unittest.TestCase):

    def test_create_folder(self):
        folder = Path('test_folder')
        create = create_folders(['Casa'], folder)
        self.assertTrue(folder.exists())

    def test_remove_folder(self):
        folder = Path('test_folder')
        if folder.exists():
            create_folders([], folder)
        remove_folder(folder)
        self.assertFalse(folder.exists())

