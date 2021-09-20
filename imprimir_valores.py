import matplotlib.pyplot as plt
import pandas as pd

history = pd.read_csv("outputs\\VGG19\\history\\resnet_val_loss_0.20.csv")

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Épocas')
plt.legend(['Treino','Validação'])
plt.grid(True)
plt.show()

# plt.figure()
# plt.plot(history['f1'])
# plt.plot(history['val_f1'])
# plt.title('F1 Score do Modelo')
# plt.ylabel('F1 Score')
# plt.xlabel('Épocas')
# plt.legend(['Treino','Validação'])
# plt.show()

plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Erro do Modelo')
plt.ylabel('Erro')
plt.xlabel('Épocas')
plt.legend(['Treino','Validação'])
plt.grid(True)
plt.show()