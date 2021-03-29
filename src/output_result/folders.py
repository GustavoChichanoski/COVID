from os.path import join, exists
from os import mkdir
from pathlib import Path
import os
from typing import List, Tuple


def create_folders(nets: List[str],
                   parent: str = './',
                   name: str = 'output') -> Tuple[List[Path],
                                                  List[Path],
                                                  List[Path]]:
    folder = Path(parent) / name
    if not exists(path=folder):
        mkdir(folder)
    nets_path, weights, figures = [], [], []
    for net in nets:
        net_path = folder / net
        weight = net_path / 'weights'
        figure = net_path / 'figures'
        if not exists(net_path):
            mkdir(net_path)
        if not exists(weight):
            mkdir(weight)
        if not exists(weight):
            mkdir(weight)
        nets_path.append(net_path)
        weights.append(weight)
        figures.append(figure)
    return nets_path, weights, figures
