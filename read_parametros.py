from typing import List
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pathlib import Path

def legenda_correta_parametros(parameter:str, label: str) -> str:
  if parameter == 'n_cuts':
    return 'Número de Pacotes'
  
  output = ""
  trueFalse(parameter, output)
  positiveNegative(parameter, output)
  
  if 'precision' in parameter:
    output += 'Precisão '
  
  if 'accuraccy' in parameter:
    output += 'Acurácia '

  if 'recall' in parameter:
    output += 'Revocação '

  if 'especifity' in parameter:
    output += 'Especificidade'
  
  if 'sensibility' in parameter:
    output += 'Sensibildade '

  return f'{parameter} para {label}.'

def positiveNegative(parameter: str, output: str) -> str:
    if 'positive' in parameter:
      output += 'Positivos '
    elif 'negative' in parameter:
      output += 'Negativos '
    return output

def trueFalse(parameter: str, output: str) -> str:
    if 'true' in parameter:
      output += ' Verdadeiros '
    elif 'false' in parameter:
      output += ' Falsos '
    return output

def plot_parameter(
  rede: str,
  x: List[float],
  covid: pd.Series,
  normal: pd.Series,
  pneumonia: pd.Series,
  parameter: str
) -> None:
  N_IMAGENS = [1,2,3,4,5,10,20,30,50,60,70,100]
  plt.figure(rede)
  plt.ylabel(f"{parameter} [%]")
  plt.xlabel('Número de Cortes')
  plt.plot(x.values, covid.values * 100)
  plt.plot(x.values, normal.values * 100)
  plt.plot(x.values, pneumonia.values * 100)
  plt.legend(['Covid','Normal','Pneumonia'])
  plt.xticks(ticks=N_IMAGENS, fontsize=10, alpha=.7)
  plt.yticks(ticks=np.arange(87,101,1), fontsize=10, alpha=.7)
  plt.grid(True)
  plt.xscale('log')
  # plt.savefig(output / 'figures' / f'{rede}_parametros.pgf')
  plt.show()
  return None

def plot_history(
  rede: str,
  df: pd.DataFrame,
  parameter: str,
  path: Path,
  real_parameter: str,
  pgf: bool = False
) -> None:
  plt.figure(rede)
  plt.ylabel(f"{real_parameter}")
  plt.xlabel('Epócas')
  treino = df[parameter]
  validacao = df[f'val_{parameter}']
  plt.plot(treino.values)
  plt.plot(validacao.values)
  plt.legend(['Treino', 'Validação'])
  plt.grid(True)
  if pgf:
    plt.savefig(path / 'figures' / f'{rede}_history.pgf', backend='pgf')
  else:
    plt.show()
  return None

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

REDES = ["ResNet50V2", "InceptionResNetV2", "VGG19", "DenseNet121"]

rede = REDES[0]
output = Path('outputs') / rede

history = output / 'history' / 'history.csv'

df = pd.read_csv(history)

plot_history(rede, df, path=output, parameter='loss', real_parameter='Erro', pgf=False)
plot_history(rede, df, path=output, parameter='accuracy', real_parameter='Acurácia', pgf=False)
csv_path = output / 'parametros.csv'

df = pd.read_csv(csv_path)

x = df["n_cuts"]

def idontcare(parameter: str, real_name: str, rede: str, df: pd.DataFrame, x: pd.Series) -> None:
  covid = df[f'covid_{parameter}']
  normal = df[f'normal_{parameter}']
  pneumonia = df[f'pneumonia_{parameter}']

  plot_parameter(
    rede=rede,
    x=x,
    covid=pneumonia,
    normal=normal,
    pneumonia=covid,
    parameter=real_name
  )

tp = df['covid_true_positive']
tn = df['covid_true_negative']

idontcare('precision', 'Precisão', rede, df, x)
idontcare('accuraccy', 'Acurácia', rede, df, x)
idontcare('recall', 'Revocação', rede, df, x)
idontcare('especifity', 'Especifidade', rede, df, x)
idontcare('sensibility', 'Sensibilidade', rede, df, x)