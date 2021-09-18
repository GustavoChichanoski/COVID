from os.path import getctime
from pathlib import Path

from tensorflow.python.keras.callbacks import History
from src.prints.prints import print_info
from typing import Any, List, Tuple, Union
from zipfile import ZipFile
import pandas as pd

def last_file(path: Path, suffix_file: str = '.hdf5') -> Path:
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

def get_all_files_in_folder(path: Path) -> List[Path]:
    paths = []
    for file in path.iterdir():
        if file.is_dir():
            paths.extend(get_all_files_in_folder(file))
        else:
            paths.append(file)
    return paths

def zip_folder(
    path: Path = Path('./outputs'),
    output_name: str = 'outputs.zip'
) -> None:
    files = get_all_files_in_folder(path)
    with ZipFile(output_name,'w') as zip:
        for file in files:
            print(file)
            zip.write(file)
    if Path(output_name).exists():
        print('[INFO] All files zipped successfully')
        return None
    print('[ERRO] Error when zipped files')
    return None

def pandas2csv(history: History, history_path: str = './history') -> None:
    """ Arquivo para salvar o treinamento do modelo em um csv.

        Args:
            history (Any): historico do treinamento do modelo.
            history_path (Path): nome do arquivo a ser salvo

        Raises:
            ValueError: Aparece quando o nome o arquivo foi corrompido

        Returns:
            None : função sem retorno
    """
    file_history = f"{history_path}.csv"
    hist_df = pd.DataFrame(history.history)
    with open(file_history, mode="w") as f:
        hist_df.to_csv(f)
        f.close()
    print_info(f'Historico salvo em {file_history}')
    return None