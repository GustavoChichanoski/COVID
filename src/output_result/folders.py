from os import mkdir
from pathlib import Path
from typing import List, Tuple, Union

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
