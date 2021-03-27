import pandas as pd
import os
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
                overwrite: bool = True) -> str:
    path_file = os.path.join(path, add_csv(name))
    i = 0
    data = pd.DataFrame.from_dict(value)
    compression_opts = dict(method='zip',
                            archive_name='out.csv')
    while os.path.exists(path_file) and not overwrite:
        rename = '{}_{}'.format(name, i)
        path_file = add_csv(rename)
        i += i
    data.to_csv(path_file, index=False)
    if verbose:
        print('Valor salvo no arquivo: `{}'.format(path_file))
    return path_file
