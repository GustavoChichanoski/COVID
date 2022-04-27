import csv
import os.path
from typing import Any, List
import numpy as np
import numpy.random as rd


def add_csv(file: str) -> str:
    return '{}.csv'.format(file)


def save_as_csv(value: Any,
                labels: List[str],
                path: str = './',
                name: str = 'matrix',
                verbose: bool = False,
                overwrite: bool = False) -> str:
    path_file = os.path.join(path, add_csv(name))
    i = 0
    while os.path_file.exist(path_file) and not overwrite:
        rename = '{}_{}'.format(name, i)
        path_file = add_csv(rename)
        i += i
    with open(path_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(labels)
        csv_writer.writerows(value)
    if verbose:
        print('Valor salvo no arquivo: `{}'.format(path_file))
    return path_file
