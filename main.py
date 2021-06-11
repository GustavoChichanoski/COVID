from pathlib import Path
from src.dataset.generator import DataGenerator
from src.plots.history import plot_history
from src.dataset.dataset import Dataset
from src.output_result.folders import create_folders, last_file
from src.model.model import ModelCovid
from varname import varname
import sys

NET = ["DenseNet201", "InceptionResNetV2", "ResNet50V2", "VGG19"]

def check_model_name(args: str):
    if args in NET:
        model_name = args
    else:
        try:
            model_name = NET[int(args)]
        except ValueError:
            raise ValueError(f'O valor {args} do modelo precisar ser um valor valido \
                               como por exemplo: {NET} , ou 0, 1, 2 e 3')
    return model_name

def check_path(arg: str):
    try:
        data = Path(arg)
        if data.exists():
            return data
        create_folders(arg)
        return data
    except ValueError:
        print(f'O caminho {arg} não é um caminho valido. \
                Exemplo de caminho válido `data`')
        return None

def check_int(dim: str):
    try:
        shape =  int(dim)
        return shape

    except ValueError:
        raise ValueError(f'[ERROR] {dim} não é um valor válido. \
                           Exemplo de um valor valido seria: `100`.')

def check_bool(value: bool):
    try:
        return bool(value)
    except ValueError:
        raise ValueError(f'[ERROR] {value} não é um valor válido. \
                            Exemplo de um valor valido seria: `True`.')

def classification_lung_image(
    output_path: Path,
    data_path: Path,
    model_name: str,
    dim: int = 224,
    epochs: int = 100,
    batch_size: int = 32,
    k_split: int = 100,
    tamanho_maximo: int = None,
    kaggle: bool = False,
    imagem_test: Path = None,
    gerar_matrix_confusao: bool = True
) -> None:
    # Variaveis utilizadas
    shape = (dim,dim,1)
    net_path = output_path / model_name
    labels = [folder.name for folder in data_path.iterdir()]

    # Folders de utilidades
    path_weight = net_path / "weights" # Caminho dos pesos
    path_figure = net_path / "figures" # Caminho para as figuras
    path_history = net_path / "history" # Caminho para o historico

    # Caminho para ultimo peso salvo
    weight = last_file(path_weight)
    
    # Criação do dataset
    ds_train = Dataset(path_data=data_path / 'train', train=False)
    ds_test = Dataset(path_data=data_path / 'test', train=False)

    # Partição dos datasets
    part_param = {"tamanho": tamanho_maximo}
    train, validation = ds_train.partition(val_size=0.2, **part_param)
    test_values, _test_val_v = ds_test.partition(val_size=1e-5, **part_param)

    # Criação dos geradores
    params = {"dim": dim, "batch_size": batch_size, "n_class": len(labels), "channels": 1}
    train_generator = DataGenerator(x_set=train[0], y_set=train[1], **params)
    val_generator = DataGenerator(x_set=validation[0], y_set=validation[1], **params)
    test_generator = DataGenerator(x_set=test_values[0], y_set=test_values[1], **params)

    # Criação do modelo
    model_params = {"labels": labels, "model_name": model_name, "model_input_shape": shape}
    covid = ModelCovid(weight_path=".model/weights.best.hfd5", **model_params)
    covid.compile(loss="categorical_crossentropy", lr=1e-5)

    if weight is not None:
        print(f"[INFO] Carregando o modelo: {weight}")
        covid.load(weight)
    else:
        fit_params = {
            "epochs": epochs, "shuffle": True, "workers": 1,
            "batch_size": batch_size, "verbose": True,
        }
        history = covid.fit_generator(
            train_generator=train_generator,
            val_generator=val_generator, **fit_params
        )
        file_model, file_weights, file_history = covid.save(
            path=net_path, history=history.history,
            metric="accuracy", kaggle=kaggle
        )
        plot_history(history)

    name = path_figure / f"{model_name}_{k_split}"
    print(f"[INFO] Predição de uma imagem: {k_split}")
    if imagem_test is not None:
        print(covid.predict(image=imagem_test, n_splits=k_split, name=name, grad=False))

    if gerar_matrix_confusao:
        matrix = covid.confusion_matrix(test_generator.x, 4)
        plot_dataset(absolut=matrix, names=labels, n_images=1, path=path_figure)

def main(args):
    model_name = NET[0]
    cwd = Path('./')
    cla_params = {
        'output_path': cwd / 'outputs', 'data_path': cwd / 'data',
        'model_name': model_name, 'dim': 224, 'epochs': 100, 'batch_size': 32,
        'k_split': 100, 'tamanho_maximo': None, 'kaggle': False,
        'imagem_test': cwd / 'data' / 'test' / '0000.png', 'gerar_matrix_confusao': True
    }
    if len(args) > 2:
        for id, arg in enumerate(args):
            if '--' in arg:
                arg = arg.split['--']
                if arg == 'model':  
                    cla_params['model_name'] = check_model_name(args[id + 1])
                if type(cla_params[arg]) == int:
                    cla_params[arg] = check_int(args[id + 1])
                if type(cla_params[arg]) == Path:
                    cla_params[arg] = check_path(args[id + 1])
                if type(cla_params[arg]) == bool:
                    cla_params[arg] = check_bool(args[id + 1])
    classification_lung_image(**cla_params)

if __name__ == '__main__':
    main(sys.argv[1:])