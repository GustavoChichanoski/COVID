"""
    $Module GradCam
"""
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import Input
from tensorflow.python.keras import backend as K
from src.images.process_images import resize_image as resize
from src.images.process_images import relu as relu_img
from numba import jit
from varname import nameof
from tqdm import tqdm


def prob_grad_cam(
    pacotes_imagem,
    classifier: List[str],
    last_conv_layer_name: str,
    posicoes_iniciais_dos_pacotes,
    modelo: Model,
    dim_orig: int = 1024,
    winner_pos: int = 0,
):
    """Gera o grad cam a partir de pedaços da imagem original
        Args:
        -----
            pacotes_imagem (array):
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
    dimensao_imagem = modelo.input_shape[1]
    dimensao_original = (dim_orig, dim_orig)
    # Inicializa a imagem final do grad cam com zeros
    grad_cam_prob = np.zeros(dimensao_original)
    # Armazena o numero de pacotes que passaram por um pixel
    pacotes_por_pixel = np.zeros(dimensao_original)
    # Recebe o tamanho da imagem a ser inserida no modelo
    grad_cam, resnet = modelo_grad_cam(modelo, last_conv_layer_name, dimensao_imagem)
    # Cria o modelo com as camadas após a ultima convolução
    model_after_last_conv = model_after(grad_cam, resnet, modelo, classifier)
    predicoes = modelo.predict(pacotes_imagem)
    entrada_modelo = modelo.input_shape
    shape = (1, entrada_modelo[1], entrada_modelo[2], entrada_modelo[3])
    for predicao, pacote_atual, posicao_pixel in tqdm(zip(
        predicoes, pacotes_imagem, posicoes_iniciais_dos_pacotes
    )):
        # Os pacotes chegam com dimensões (None,224,224,3) e
        # precisam ser redimensionados para a (1,224,224,3)
        pacote_atual_reshape = pacote_atual.reshape(shape)
        # Define a predição do canal vencedor geral
        predicao_pacote = predicao[winner_pos]
        # Calcula o grad cam para o canal vencedor
        last_conv_layer_output = generate_grads_image(
            grad_cam, pacote_atual_reshape, model_after_last_conv
        )
        gradcam_pacote = create_heatmap(last_conv_layer_output, dimensao_imagem)
        # Multiplica a possibilidade do canal pelo grad cam gerado
        grad_cam_predicao = predicao_pacote * gradcam_pacote
        # Soma com a predicoes anteriores
        somar_resultado = somar_grads_cam(
            grad_cam_prob, grad_cam_predicao, pacotes_por_pixel, posicao_pixel
        )
        grad_cam_prob, pacotes_por_pixel = somar_resultado
    # Divide o gradcam total pelas vezes que passou por um pixel
    div_grad_cam_prob = divisao_pacotes_por_pixel(pacotes_por_pixel, grad_cam_prob)
    grad_cam_prob = normalize(div_grad_cam_prob)
    return grad_cam_prob

def modelo_grad_cam(modelo, last_conv_layer_name, dimensao_imagem):
    # Recebe o tamanho da imagem a ser inserida no modelo
    resnet = find_base_model(modelo)
    # Nome da ultima camada de convolucao
    last_conv_layer = get_layer(resnet, last_conv_layer_name)
    # Cria um novo modelo com as camadas até a ultima convolução
    model_until_last_conv = Model(resnet.input, last_conv_layer.output)
    grad_cam_input = (dimensao_imagem, dimensao_imagem, modelo.input_shape[3])
    inputs = Input(shape=grad_cam_input)
    new_model = modelo.layers[1](inputs)
    new_model = model_until_last_conv(new_model)
    new_model = Model(inputs, new_model)
    return new_model, resnet

def get_layer(resnet: Model, last_conv_layer_name: str):
    for layer in reversed(resnet.layers):
        if layer.name == last_conv_layer_name:
            return layer


@jit(nopython=True)
def somar_grads_cam(
    grad_cam_full,
    grad_cam_cut,
    used_pixels,
    start: Tuple[int,int] = (0, 0),
):
    """
    Realiza a soma do GradCam completo com o GradCam do recorte.
    Para isso é necessário conhecer os valores atuais da GradCam,
    a GradCam gerada pelo recorte da imagem e as posicões inicial
    do recorte.

    Args:
    -----
        grad_cam_atual (np.array):
            GradCam atual da imagem
        grad_cam_split (np.array):
            GradCam do recorte da imagem
        used_pixels (np.array):
            Matriz de pacotes por pixels.
        start (Tuple[int,int], optional):
            Posições iniciais dos recortes.
            Defaults to ```(0, 0)```.

    Returns:
    --------
        (tuple): GradCam calculada até o momento, pacotes por Pixel

    Raises:
    --------
        ValueError:
            if grad_cam_full and used_pixels don't have different shapes.
        ValueError:
            if grad_cam_cut have one dimension bigger than grad_cam_full.
    """
    # if grad_cam_full.shape == used_pixels.shape:
    #     raise ValueError(
    #         "The shape of {} and {} need be the same \n \
    #             Shape of {}: {} \n \
    #             Shape of {}: {} \n \
    #         ".format(
    #         nameof(grad_cam_full), nameof(used_pixels)
    #     ))

    # for dimension_full, dimension_cut in zip(grad_cam_full.shape,
    #                                          grad_cam_cut.shape):
    #     if dimension_full < dimension_cut:
    #         raise ValueError(
    #             "The shape of {} must be equal or lesser than {}, \
    #             \n Shape of {}: {} \
    #             \n Shape of {}: {}".format(
    #                 nameof(grad_cam_cut),nameof(grad_cam_full),
    #                 grad_cam_full.shape, grad_cam_cut.shape
    #             )
    #         )
    # Get the dimension of cut grad_cam
    dimension_cut = grad_cam_cut.shape[0]
    # Create the matriz of ones to sum to grad cam full
    ones = np.ones((dimension_cut, dimension_cut))
    final = [start[0] + dimension_cut, start[1] + dimension_cut]
    grad_cam_full[start[0] : final[0], start[1] : final[1]] += grad_cam_cut
    used_pixels[start[0] : final[0], start[1] : final[1]] += ones
    return (grad_cam_full, used_pixels)


@jit(nopython=True)
def divisao_pacotes_por_pixel(pacotes_por_pixel: List[int], grad_cam_prob: List[int]):
    """Divide os numeros de pacotes de pixeis pela
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
            pacotes = pacotes_por_pixel[pixel_y][pixel_x]
            if pacotes > 0:
                grad_cam_prob[pixel_y][pixel_x] /= pacotes
    return grad_cam_prob


def find_base_model(model):
    for layer in model.layers:
        if type(model) == type(layer):
            return layer


def grad_cam(
    image,
    model,
    classifier_layer_names: List[str],
    last_conv_layer_name: str = "avg_pool",
):
    """ Gera o mapa de processamento da CNN.
        Args:
            image (np.array): Recorte da imagem
            model (keras.Model): Modelo da CNN
        Returns:
            (np.array): Grad Cam
    """
    dimensao_imagem = model.input_shape[1]
    # Altera o dimensão da imagem para atender a entrada do modelo
    dimensao_modelo = (1, dimensao_imagem, dimensao_imagem, model.input_shape[3])
    image_reshape = image.reshape(dimensao_modelo)
    # Recebe o tamanho da imagem a ser inserida no modelo
    resnet = find_base_model(model)
    # Nome da ultima camada de convolucao
    last_conv_layer = resnet.get_layer(last_conv_layer_name)
    # Cria um novo modelo com as camadas até a ultima convolução
    model_until_last_conv = Model(resnet.input, last_conv_layer.output)
    inputs = Input(shape=(dimensao_imagem, dimensao_imagem, model.input_shape[3]))
    new_model = model.layers[1](inputs)
    new_model = model_until_last_conv(new_model)
    new_model = Model(inputs, new_model)
    # Cria o modelo com as camadas após a ultima convolução
    model_after_last_conv = model_after(
        new_model, resnet, model, classifier_layer_names
    )
    last_conv_layer_output = generate_grads_image(
        new_model, image_reshape, model_after_last_conv
    )
    heatmap = create_heatmap(last_conv_layer_output, dimensao_imagem)
    return heatmap


def generate_grads_image(model_until_last_conv, image, classifier_model):
    """Compute the gradient of the top predicted
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

    heatmap = grads_calc(tape, last_cnn_output, top_class_channel)
    return heatmap


def grads_calc(
    tape: tf.GradientTape, last_cnn_output: Model, top_class_channel: List[List[int]]
):
    """"""
    # Calcula o gradiente do modelo para saída máxima (pesos)
    grads = tape.gradient(top_class_channel, last_cnn_output)
    # Retira pela media dos pixel (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_cnn_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    return tf.squeeze(heatmap)
    # # Convert array to numpy array
    # pesos = last_cnn_output.numpy()[0]
    # pooled_grads = pooled_grads.numpy()
    # # Multiplica pesos pela importância deles no resultados
    # for i in range(pooled_grads.shape[-1]):
    #     pesos[:, :, i] *= pooled_grads[i]
    # # pooled grads == ack
    # return pesos


def model_after(
    last_conv_layer, resnet_model, model, classifier_layer_names: List[str]
):
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


def create_heatmap(last_conv_layer_output, dimensao_saida: int = 224):
    """Cria o heatmap do recorte da imagem gerada
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
    # heatmap = np.mean(last_conv_layer_output, axis=-1)
    # Normalização do heatmap
    heatmap = np.array(last_conv_layer_output)
    heatmap = relu_img(heatmap)
    heatmap = resize(heatmap, dimensao_saida)
    return heatmap


def normalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == 0 and x_min == 0:
        return x
    x_new = (x - x_min) / (x_max - x_min)
    return x_new