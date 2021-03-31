import pandas as pd
import os
from typing import Any
from pathlib import Path

def save_as_csv(value: Any,
                path: Path = Path('./'),
                name: Path = Path('matrix'),
                verbose: bool = False,
                overwrite: bool = True) -> Path:

    path_file = path / f'{name}.csv'
    data = pd.DataFrame.from_dict(value)

    i = 0
    while os.path.exists(path_file) and not overwrite:
        path_file = path / f'{name}_{i}.csv'
        i += 1

    data.to_csv(path_file, index=False)

    if verbose:
        print(f'Valor salvo no arquivo: {path_file}')

    return path_file
