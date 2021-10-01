import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from src.plots.save_as_png import save_png

REDES = ["ResNet50V2", "InceptionResNetV2", "VGG19", "DenseNet121"]
NET = REDES[3]

output = Path("outputs")
output = output / NET

history_path = output / "history" / "history.csv"
history = pd.read_csv(history_path)

epochs = history["epochs"]
max_epoch = epochs.values[-1]
step = np.arange(0,max_epoch, max_epoch / 10)

fig = plt.figure()
plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])
plt.title("Acurácia do Modelo")
plt.ylabel("Acurácia")
plt.xlabel("Épocas")
plt.legend(["Treino", "Validação"])
plt.grid(True)
# save_png(fig=fig, path=output / "figures" / "acuracia.png", overwrite=True)
plt.show()

# plt.figure()
# plt.plot(history['f1'])
# plt.plot(history['val_f1'])
# plt.title('F1 Score do Modelo')
# plt.ylabel('F1 Score')
# plt.xlabel('Épocas')
# plt.legend(['Treino','Validação'])
# plt.show()

fig = plt.figure()
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("Erro do Modelo")
plt.ylabel("Erro")
plt.xlabel("Épocas")
plt.legend(["Treino", "Validação"])
plt.grid(True)
save_png(fig=fig, path=output / "figures" / "erro.png", overwrite=True)
plt.show()