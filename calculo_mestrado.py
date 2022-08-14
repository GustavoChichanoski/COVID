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

  p = (tp / (tp + fp)) * 100
  a = ((tp + tn) / (tp + tn + fp + fn)) * 100
  r = ((tp) / (tp + fn)) * 100
  e = (tn / (tn + fn)) * 100
  s = (tp / (tp + fn)) * 100

  print(f"[tp: {tp}, fp: {fp}, pp: {tp + fp}]")
  print(f"[fn: {fn}, tn: {tn}, pn: {tn + fn}]")
  print(f"[rp: {tp + fn}, rn: {fp + tn}, rt: {tp + fn + fp + tn} ]")
  print(f"p: {p:0.2f}")
  print(f"a: {a:0.2f}")
  print(f"r: {r:0.2f}")
  print(f"e: {e:0.2f}")
  print(f"s: {s:0.2f}")

def print_linha():
  print("-----------------------")

resnet_mc_5 = np.array([[268,1,13],[4, 507, 40],[4, 30, 839]])
desnet_mc_5 = np.array([[267,1,14],[3, 502, 46],[1, 27, 845]])
desnet_mc_100 = np.array([[273,0,9],[1,513,37],[2,23,848]])
inception_mc_5 = np.array([[269,2,11],[2,508,41],[0,31,842]])
vgg19_mc_5 = np.array([[253,4,25],[2,512,37],[5,44,824]])
densenet_169 = np.array([[252.,   2.,   0.],
                         [  3., 516.,  67.],
                         [ 27.,  33., 807.]])
densenet_201 = np.array([[256.,   5.,   4.],
                         [  6., 494.,  39.],
                         [ 20.,  52., 831.]])
resnet101v2 = np.array([[268.,   3.,   2.],
                        [  2., 515.,  38.],
                        [ 12.,  33., 834.]])
vgg16 = np.array([[239.,   1.,   5.],
                  [  5., 494.,  40.],
                  [ 38.,  56., 829.]])
print("------- ResNet --------")
valores_mestrado(resnet_mc_5, 0)
print_linha()
print("------- DenseNet ------")
valores_mestrado(desnet_mc_5, 0)
print_linha()
print("----- DenseNet121 -----")
valores_mestrado(desnet_mc_100, 0)
print_linha()
print("------ Inception ------")
valores_mestrado(inception_mc_5, 0)
print_linha()
print("------ VGG19 ----------")
valores_mestrado(vgg19_mc_5, 0)
print_linha()
print("------ DenseNet169 ----------")
valores_mestrado(densenet_169, 0)
print_linha()
print("------ DenseNet201 ----------")
valores_mestrado(densenet_201, 0)
print_linha()
print("------ ResNet101v2 ----------")
valores_mestrado(resnet101v2, 0)
print_linha()
print("------ VGG-16 ----------")
valores_mestrado(vgg16, 0)
print_linha()
