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

def plot_parameter( df: pd.DataFrame,x: str, parameter: str) -> None:
  labels = ['covid', 'normal', 'pneumonia']
  LABELS = ['Covid','Normal','Pneumonia']
  x = df[x]
  y0 = df[f"{labels[0]}_{parameter}"]
  y0 = df[f"{labels[1]}_{parameter}"]
  y0 = df[f"{labels[2]}_{parameter}"]

  plt.figure(rede)
  plt.ylabel(parameter)
  plt.xlabel('Número de Cortes')
  plt.plot(x.values,y.values*100)
  plt.plot(x.values,y2.values*100)
  plt.plot(x.values,y3.values*100)
  plt.legend(LABELS)
  plt.xticks(ticks=N_IMAGENS, fontsize=10, alpha=.7)
  plt.yticks(ticks=np.arange(87,101,1), fontsize=10, alpha=.7)
  plt.grid(True)
  plt.xscale('log')
  # plt.savefig(output / 'figures' / f'{rede}_parametros.pgf')
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
N_IMAGENS = [1,2,3,4,5,10,20,30,50,60,70,100]

rede = REDES[1]
output = Path('outputs') / rede

csv_path = output / 'parametros.csv'

df = pd.read_csv(csv_path)

x = df["n_cuts"]
y = df['covid_precision']
y2 = df['normal_precision']
y3 = df['pneumonia_precision']

plt.figure(rede)
plt.ylabel('Precisão')
plt.xlabel('Número de Cortes')
plt.plot(x.values,y.values*100)
plt.plot(x.values,y2.values*100)
plt.plot(x.values,y3.values*100)
plt.legend(['Covid','Normal','Pneumonia'])
plt.xticks(ticks=N_IMAGENS, fontsize=10, alpha=.7)
plt.yticks(ticks=np.arange(87,101,1), fontsize=10, alpha=.7)
plt.grid(True)
plt.xscale('log')
# plt.savefig(output / 'figures' / f'{rede}_parametros.pgf')
plt.show()

x = df["n_cuts"]
y = df['covid_true_positives']
y2 = df['normal_true_positives']
y3 = df['pneumonia_true_positives']

plt.figure(rede)
plt.ylabel('Verdadeiros Positivos')
plt.xlabel('Número de Cortes')
plt.plot(x.values,y.values*100)
plt.plot(x.values,y2.values*100)
plt.plot(x.values,y3.values*100)
plt.legend(['Covid','Normal','Pneumonia'])
plt.xticks(ticks=N_IMAGENS, fontsize=10, alpha=.7)
plt.yticks(ticks=np.arange(87,101,1), fontsize=10, alpha=.7)
plt.grid(True)
plt.xscale('log')
# plt.savefig(output / 'figures' / f'{rede}_parametros.pgf')
plt.show()

y = df['covid_accuraccy']
y2 = df['normal_accuraccy']
y3 = df['pneumonia_accuraccy']

plt.figure()
plt.ylabel('Acuracia')
plt.xlabel('Número de Cortes')
plt.plot(x.values,y.values*100)
plt.plot(x.values,y2.values*100)
plt.plot(x.values,y3.values*100)
plt.legend(['Covid','Normal','Pneumonia'])
# plt.savefig(output / 'figures' / f'{rede}_parametros.pgf')
plt.show()

y = df['covid_recall']
y2 = df['normal_recall']
y3 = df['pneumonia_recall']

plt.figure()
plt.ylabel('Revocação')
plt.xlabel('Número de Cortes')
plt.plot(x.values,y.values*100)
plt.plot(x.values,y2.values*100)
plt.plot(x.values,y3.values*100)
plt.legend(['Covid','Normal','Pneumonia'])
# plt.savefig(output / 'figures' / f'{rede}_parametros.pgf')
plt.show()

y = df['covid_especifity']
y2 = df['normal_especifity']
y3 = df['pneumonia_especifity']

plt.figure()
plt.ylabel('Especificade')
plt.xlabel('Número de Cortes')
plt.plot(x.values,y.values*100)
plt.plot(x.values,y2.values*100)
plt.plot(x.values,y3.values*100)
plt.legend(['Covid','Normal','Pneumonia'])
# plt.savefig(output / 'figures' / f'{rede}_parametros.pgf')
plt.show()

y = df['covid_sensibility']
y2 = df['normal_sensibility']
y3 = df['pneumonia_sensibility']

plt.figure()
plt.ylabel('Sensibilidade')
plt.xlabel('Número de Cortes')
plt.plot(x.values,y.values*100)
plt.plot(x.values,y2.values*100)
plt.plot(x.values,y3.values*100)
plt.legend(['Covid','Normal','Pneumonia'])
# plt.savefig(output / 'figures' / f'{rede}_parametros.pgf')
plt.show()