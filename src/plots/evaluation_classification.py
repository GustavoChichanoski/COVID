from src.dataset import generator
from typing import List
import tensorflow_addons as tfa
from tensorflow.python.keras import Model
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.classification.cla_generator import ClassificationDatasetGenerator
from src.models.classificacao.funcional_model import confusion_matrix


def true_positive(matriz: tfa.types.TensorLike, label: int) -> tfa.types.TensorLike:
    return matriz[label][label]


def true_negative(matriz: tfa.types.TensorLike, label: int) -> tfa.types.TensorLike:
    soma: int = 0
    for row in range(matriz.shape[0]):
        for column in range(matriz.shape[1]):
            if row != label and column != label:
                soma += matriz[column][row]
    return soma


def false_positive(matriz: tfa.types.TensorLike, label: int) -> tfa.types.TensorLike:
    soma: int = 0
    for column in range(matriz.shape[1]):
        if column != label:
            soma += matriz[label][column]
    return soma


def false_negative(matriz: tfa.types.TensorLike, label: int) -> tfa.types.TensorLike:
    soma: int = 0
    for row in range(matriz.shape[0]):
        if row != label:
            soma += matriz[row][label]
    return soma


def precision(tp: float, tn: float, fp: float, fn: float) -> float:
    return tp / (tp + fp) if (tp + fp) != 0 else 1


def accuracy(tp: float, tn: float, fp: float, fn: float) -> float:
    den = (tp + tn + fp + fn)
    return (tp + tn) / den if den != 0 else 1


def recall(tp: float, tn: float, fp: float, fn: float) -> float:
    return tp / (tp + fn) if (tp + fn) != 0 else 1


def especifity(tp: float, tn: float, fp: float, fn: float) -> float:
    return tn / (tn + fn) if (tn + fn) != 0 else 1


def sensibility(tp: float, tn: float, fp: float, fn: float) -> float:
    return tp / (tp + fn) if (tp + fn) != 0 else 1


def plot_mc_in_csv(
    model: Model,
    test_generator: ClassificationDatasetGenerator,
    path_to_save: Path,
    n_imagens: List[int]
) -> None:
    colunas = np.array([])
    colums_name = [
        "covid_true_positive",
        "covid_true_negative",
        "covid_false_positive",
        "covid_false_negative",
        "covid_precision",
        "covid_accuraccy",
        "covid_recall",
        "covid_especifity",
        "covid_sensibility",
        "normal_true_positive",
        "normal_true_negative",
        "normal_false_positive",
        "normal_false_negative",
        "normal_precision",
        "normal_accuraccy",
        "normal_recall",
        "normal_especifity",
        "normal_sensibility",
        "pneumonia_true_positive",
        "pneumonia_true_negative",
        "pneumonia_false_positive",
        "pneumonia_false_negative",
        "pneumonia_precision",
        "pneumonia_accuraccy",
        "pneumonia_recall",
        "pneumonia_especifity",
        "pneumonia_sensibility",
        "n_cuts",
    ]
    for cuts in n_imagens:
        matriz0 = confusion_matrix(model, test_generator, cuts)
        matriz1 = confusion_matrix(model, test_generator, cuts)
        matriz2 = confusion_matrix(model, test_generator, cuts)
        linha = np.array(cuts)
        for label in range(3):
            p0 = calculate_parameters_evaluation(matriz0, label)
            p1 = calculate_parameters_evaluation(matriz1, label)
            p2 = calculate_parameters_evaluation(matriz2, label)
            parameters = np.zeros((len(p0)))
            for i in range(len(p0)):
                parameters[i] = (p0[i] + p1[i] + p2[i]) / 3
            linha = np.append(parameters, linha)
        colunas = np.append(linha, colunas)
    colunas = np.reshape(colunas, (len(n_imagens), len(colums_name)))
    df = pd.DataFrame(data=colunas, columns=colums_name)
    df.index.name='index'
    df.to_csv(path_to_save,index=True, index_label='Index')


def calculate_parameters_evaluation(
    matriz: tfa.types.TensorLike,
    label: int
) -> tfa.types.TensorLike:
    linha = np.array([])
    tp = true_positive(matriz, label)
    tn = true_negative(matriz, label)
    fp = false_positive(matriz, label)
    fn = false_negative(matriz, label)
    p = precision(tp=tp, tn=tn, fp=fp, fn=fn)
    a = accuracy(tp=tp, tn=tn, fp=fp, fn=fn)
    r = recall(tp=tp, tn=tn, fp=fp, fn=fn)
    e = especifity(tp=tp, tn=tn, fp=fp, fn=fn)
    s = sensibility(tp=tp, tn=tn, fp=fp, fn=fn)
    linha = np.append(linha, tp)
    linha = np.append(linha, tn)
    linha = np.append(linha, fp)
    linha = np.append(linha, fn)
    linha = np.append(linha, p)
    linha = np.append(linha, a)
    linha = np.append(linha, r)
    linha = np.append(linha, e)
    linha = np.append(linha, s)
    return linha