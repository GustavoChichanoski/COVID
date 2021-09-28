import numpy as np

def true_positive(matriz, label: int):
  return matriz[label][label]

def true_negative(matriz, label):
  soma: int = 0
  for row in range(matriz.shape[0]):
    for column in range(matriz.shape[1]):
      if row != label and column != label:
        soma += matriz[column][row]
  return soma

def false_positive(matriz, label: int):
  soma: int = 0
  for column in range(matriz.shape[1]):
    if column != label:
      soma += matriz[label][column]
  return soma

def false_negative(matriz, label: int):
  soma : int = 0
  for row in range(matriz.shape[0]):
    if row != label:
      soma += matriz[row][label]
  return soma

def valores_mestrado(matriz, label):
  tp = true_positive(matriz, label)
  tn = true_negative(matriz, label)
  fp = false_positive(matriz, label)
  fn = false_negative(matriz, label)

  p = tp / (tp + fp)
  a = ((tp + tn) / (tp + tn + fp + fn))
  r = ((tp) / (tp + fn))
  e = (tn / (tn + fn))
  s = tp / (tp + fn)

  print(f"[tp: {tp}, fp: {fp}, pp: {tp + fp}]")
  print(f"[fn: {fn}, tn: {tn}, pn: {tn + fn}]")
  print(f"[rp: {tp + fn}, rn: {fp + tn}, rt: {tp + fn + fp + tn} ]")
  print(f"p: {p}")
  print(f"a: {a}")
  print(f"r: {r}")
  print(f"e: {e}")
  print(f"s: {s}")

def print_linha():
  print("-----------------------")

resnet_mc_5 = np.array([[268,1,13],[4, 507, 40],[4, 30, 839]])
desnet_mc_5 = np.array([[267,1,14],[3, 502, 46],[1, 27, 845]])
desnet_mc_100 = np.array([[273,0,9],[1,513,37],[2,23,848]])
inception_mc_5 = np.array([[269,2,11],[2,508,41],[0,31,842]])
vgg19_mc_5 = np.array([[253,4,25],[2,512,37],[5,44,824]])

print("------- ResNet --------")
valores_mestrado(resnet_mc_5, 0)
print_linha()
print("------- DenseNet ------")
valores_mestrado(desnet_mc_5, 0)
print_linha()
print("----- DenseNet100 -----")
valores_mestrado(desnet_mc_100, 0)
print_linha()
print("------ Inception ------")
valores_mestrado(inception_mc_5, 0)
print_linha()
print("------ VGG19 ----------")
valores_mestrado(vgg19_mc_5, 0)
print_linha()
