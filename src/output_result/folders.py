from os import mkdir
from os.path import getctime
from pathlib import Path
from typing import List, Tuple, Union

def last_file(path: Path,suffix_file: str = 'hdf5'):
    weight = None
    max_weight = None
    for weight in path.iterdir():
        suffix = weight.suffix
        if suffix == suffix_file:
            if max_weight is None:
                max_weight = weight
            else:
                time_max = getctime(max_weight)
                time_weight = getctime(weight)
                if time_max < time_weight:
                    max_weight = weight
    return max_weight

def remove_folder(path: Union[Path,List[Path]]) -> None:
    if isinstance(path,list):
        for p in path:
            remove_folder(p)
        return None
    elif path.exists():
        for child in path.iterdir():
            if child.is_dir():
                remove_folder(child)
            else:
                child.unlink()
        path.rmdir()
    return None

def create_folders(
    nets: List[str],
    name: Path = Path('output')
) -> Tuple[List[Path], List[Path], List[Path]]:

    folder = name
    folder.mkdir(exist_ok=True)

    nets_path = []

    for net in nets:

        net_path = folder / net
        net_path.mkdir(exist_ok=True)
        nets_path.append(net_path)

        weight = net_path / 'weights'
        weight.mkdir(exist_ok=True)
        
        figure = net_path / 'figures'
        figure.mkdir(exist_ok=True)
        
        history = net_path / 'history'
        history.mkdir(exist_ok=True)

    return nets_path
