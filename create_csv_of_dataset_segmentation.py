from pathlib import Path
from typing import Dict, List
import random
import pandas as pd
import numpy as np


def calculate_parts(num_total_files: int,
                    test: float = 0.1,
                    valid: float = 0.2) -> List[int]:
    num_files_tests = int(num_total_files * test)
    num_files_valid = int((num_total_files - num_files_tests) * valid)
    num_files_train = num_total_files - (num_files_tests + num_files_valid)
    return np.array([num_files_train, num_files_valid, num_files_tests])


def read_path_valids(path: Path) -> Dict:

    if path.is_dir() is True:

        files_by_type = {}

        files_by_type['type'] = {}
        files_by_type['lung'] = {}
        files_by_type['mask'] = {}

        types = ['train', 'valid', 'tests']

        files = list(path.glob('lungs/*.png'))
        num_files = len(files)

        num_files_by_types = calculate_parts(num_files)

        id_dict = 0

        for num_files_of_type, type_file in zip(num_files_by_types, types):
            for file in random.sample(files, num_files_of_type):
                file_parts = list(file.parts)
                file_parts[-2] = 'masks'
                file_mask = Path(*file_parts)

                files_by_type['type'][str(id_dict)] = type_file
                files_by_type['lung'][str(id_dict)] = str(file)
                files_by_type['mask'][str(id_dict)] = str(file_mask)

                id_dict += 1
                files.remove(file)

        df_files = pd.DataFrame(files_by_type)
        df_files.to_csv(path / 'metadata_segmentation.csv')


# read_path_valids(Path('dataset\\segmentation'))
