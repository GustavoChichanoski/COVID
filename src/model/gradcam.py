"""
    $Module GradCam
"""
from typing import List
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import Input
from keras import backend as K
from src.images.process_images import resize_image as resize
from src.images.process_images import relu_image as relu_img


def prob_grad_cam(pacotes_da_imagem,
                  posicoes_iniciais_dos_pacotes,
                  modelo: Model,
                  dim_orig: int = 1024):
    """ Gera o grad cam a partir de pedaços da imagem original

        Args:
        -----
            pacotes_da_imagem (array):
                Pacotes das imagens originais
            posicoes_iniciais_dos_pacotes (array):
                Pixeis iniciais dos pacotes.
            modelo (Keras.Model):
                Modelo desenvolvido
            dim_orig_da_imagem (int, optional):
                Dimensão original da imagem. Defaults to 1024.

        Returns:
        --------
            (np.array): Grad Cam dos recortes
    """
    grad_cam_prob = np.zeros((dim_orig, dim_orig))
    pacotes_por_pixel = np.zeros((dim_orig, dim_orig))
    for pacote_atual, posicao_pixel in zip(pacotes_da_imagem,
                                           posicoes_iniciais_dos_pacotes):
        entrada_modelo = modelo.input_shape
        pacote_atual = pacote_atual.reshape((1,
                                             entrada_modelo[1],
                                             entrada_modelo[2],
                                             entrada_modelo[3]))
        predicao = modelo.predict(pacote_atual)
        predicao_pacote = np.max(predicao)
        gradcam_pacote = grad_cam(pacote_atual, modelo)
        grad_cam_predicao = predicao_pacote * gradcam_pacote
        somar_resultado = somar_grads_cam(grad_cam_prob,
                                          grad_cam_predicao,
                                          pacotes_por_pixel,
                                          posicao_pixel)
        pacotes_por_pixel, grad_cam_prob = somar_resultado
    grad_cam_prob = divisao_pacotes_por_pixel(pacotes_por_pixel,
                                              grad_cam_prob)
    grad_cam_prob = relu_img(grad_cam_prob)
    return grad_cam_prob


def somar_grads_cam(grad_cam_atual: List[int],
                    grad_cam_split: List[int],
                    pixels_usados: List[int],
                    inicio: tuple = (0, 0)):
    """
        Realiza a soma do GradCam gerado pelo GradCam do
        recorte, com a matriz que será o GradCam final.
        Para isso é necessário conhecer os valores atuais
        da GradCam, a GradCam gerada pelo recorte da
        imagem e as posicões inicial do recorte.

        Args:
        -----
            grad_cam_atual (np.array):
                GradCam atual da imagem
            grad_cam_split (np.array):
                GradCam do recorte da imagem
            pixels_usados (np.array):
                Matriz de pacotes por pixels.
            inicio (tuple, optional):
                Posições iniciais dos recortes.
                Defaults to (0, 0).

        Returns:
        --------
            (tuple): GradCam calculada até o momento,
                     pacotes por Pixel
    """
    dimensao = grad_cam_split.shape[0]
    valor_um = np.ones((dimensao, dimensao))
    final = [inicio[0] + dimensao, inicio[1] + dimensao]
    grad_cam_atual[inicio[0]:final[0],
                   inicio[1]:final[1]] += grad_cam_split
    pixels_usados[inicio[0]:final[0], inicio[1]:final[1]] += valor_um
    return (grad_cam_atual, pixels_usados)


def divisao_pacotes_por_pixel(pacotes_por_pixel: List[int],
                              grad_cam_prob: List[int]):
    """ Divide os numeros de pacotes de pixeis pela
        grad cam do pacote gerado pela função para
        gerar o grad cam probabilistico.

        Args:
        -----
            pacotes_por_pixel (np.array):
                Numero dos pacotes passados em cada pixel.
            grad_cam_prob (np.array):
                GradCam do recorte da imagem.


        Returns:
        --------
            (np.array): GradCam Probabilistico
    """
    for pixel_y, _ in enumerate(grad_cam_prob):
        for pixel_x, _ in enumerate(grad_cam_prob[0]):
            pacotes = pacotes_por_pixel[pixel_x][pixel_y]
            if pacotes > 0:
                grad_cam_prob[pixel_y][pixel_x] /= pacotes
    return grad_cam_prob


def grad_cam(image, model,
             classifier_layer_names: List[str] = ['Max_5',
                                                  'Flatten',
                                                  'Dense_1',
                                                  'Drop',
                                                  'Dense_2']):
    """ Gera o mapa de processamento da CNN.

        Args:
            image (np.array): Recorte da imagem
            model (keras.Model): Modelo da CNN

        Returns:
            (np.array): Grad Cam
    """
    # Recebe o tamanho da imagem a ser inserida no modelo
    dimensao_imagem: int = model.input_shape[1]
    # Altera o dimensão da imagem para atender a entrada do modelo
    dimensao_modelo = (1, dimensao_imagem, dimensao_imagem, 3)
    image = image.reshape(dimensao_modelo)
    # Recebe a modelo da Resnet50-v2
    resnet_model = model.layers[0]
    # Nome da ultima camada de convolucao
    last_conv_layer = resnet_model.get_layer('post_relu')
    # Cria um novo modelo com as camadas até a ultima convolução
    model_until_last_conv = Model(resnet_model.input,
                                  last_conv_layer.output)
    # Cria o modelo com as camadas após a ultima convolução
    model_after_last_conv = model_after(last_conv_layer,
                                        resnet_model,
                                        model,
                                        classifier_layer_names)

    last_conv_layer_output = generate_grads_image(model_until_last_conv,
                                                  image,
                                                  model_after_last_conv)
    heatmap = create_heatmap(last_conv_layer_output, dimensao_imagem)
    return heatmap


def generate_grads_image(model_until_last_conv,
                         image,
                         classifier_model):
    """ Compute the gradient of the top predicted
        class for our input image with respect
        to the activations of the last conv layer

        Args:
        -----
            last_conv_layer_model (keras.layers):
                Ultima camada de convolução do modelo
            image (np.array):
                Imagem original
            classifier_model (keras.Model):
                Modelo de classificação sem convolução

        Returns:
        --------
            np.array: GradCam gerado
    """
    # Importa GradriantTape como tape
    with tf.GradientTape() as tape:
        # Recebe a saída do modelo até a ultima cadada
        # tendo como entrada a imagem
        last_cnn_output = model_until_last_conv(image)
        # Define o modelo que o tape tem que assistir
        tape.watch(last_cnn_output)
        # Calcula a predicao da camada de saída
        preds = classifier_model(last_cnn_output)
        # Recebe o index da maior predicao do sistema
        top_pred_index = tf.argmax(preds[0])
        # Recebe todas as predições da predicao ganhadora
        top_class_channel = preds[:, top_pred_index]

    last_cnn_output, pooled_grads = grads_calc(tape,
                                               last_cnn_output,
                                               top_class_channel)
    for i in range(pooled_grads.shape[-1]):
        last_cnn_output[:, :, i] *= pooled_grads[i]
    return last_cnn_output


def grads_calc(tape: tf.GradientTape,
               last_cnn_output: Model,
               top_class_channel: List[List[int]]):
    # Calcula o gradient do modelo para saída máxima
    grads = tape.gradient(top_class_channel,
                          last_cnn_output)
    # Multiplica pesos pela importância deles no resultados
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Convert array to numpy array
    last_cnn_output = last_cnn_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    return last_cnn_output, pooled_grads


def model_after(last_conv_layer,
                resnet_model,
                model,
                classifier_layer_names: List[str]):
    """Cria apenas o modelo de classificação do modelo passado pelo usuario

        Args:
            last_conv_layer (Keras.model.layers): ultima camada de convolução (ativação)
            resnet_model (Keras.model): modelo da cnn do usuario
            model (Keras.model): modelo passado pelo usuario
            classifier_layer_names (list.Strings): Nome das camadas de classificação

        Returns:
            Keras.model: modelo de classificação do usuario

        Exemplo:
        --------
    """
    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    layer = classifier_input
    for layer_name in classifier_layer_names:
        try:
            layer = resnet_model.get_layer(layer_name)(layer)
        except ValueError:
            layer = model.get_layer(layer_name)(layer)
    classifier_model = Model(classifier_input, layer)
    return classifier_model


def create_heatmap(last_conv_layer_output,
                   dimensao_saida: int = 224):
    """ Cria o heatmap do recorte da imagem gerada

        Args:
        -----
            last_conv_layer_output (keras.layers):
                ultima convolução do modelo
            dimensao_saida (int, optional):
                Dimensão da imagem de saída.
                Defaults to 224.

        Returns:
        --------
            (np.array): imagem do heatmap gerado
    """
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heat_max = np.max(heatmap)
    heat_maximum = np.maximum(heatmap, 0)
    if heat_max == 0:
        heat_max = 1
    heatmap = heat_maximum / heat_max
    heatmap = np.array(255 * heatmap)
    heatmap = np.uint8(heatmap)
    heatmap = resize(heatmap, dimensao_saida)
    heatmap = relu_img(heatmap)
    return heatmap


def target_category_loss(x, category_index, nb_classes: int):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x: float):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
