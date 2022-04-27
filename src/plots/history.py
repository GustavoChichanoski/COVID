import matplotlib.pyplot as plt

def plot_history_key(history):
    keys = history.history.keys()
    for key in keys:
        plt.figure()
        plt.plot(history.history[key])
    return None

def plot_history(history):
    keys = history.history.keys()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Épocas')
    plt.legend(['Treino','Validação'])
    plt.show()

    plt.figure()
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('F1 Score do Modelo')
    plt.ylabel('F1 Score')
    plt.xlabel('Épocas')
    plt.legend(['Treino','Validação'])
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Erro do Modelo')
    plt.ylabel('Erro')
    plt.xlabel('Épocas')
    plt.legend(['Treino','Validação'])
    plt.show()