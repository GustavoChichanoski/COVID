from os.path import join, exists
from os import mkdir
import os
from typing import List


def create_folders(nets: List[str],
                   parent: str = './',
                   name: str = 'output') -> List[str]:
    folder = join(parent,name)
    if not exists(path=folder):
        mkdir(folder)
    nets_path, weights, figures = [], [], []
    for net in nets:
        net_path = join(folder,net)
        weight = join(net_path,'weights')
        figure = join(net_path,'figures')
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

