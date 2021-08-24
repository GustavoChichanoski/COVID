"""
    $Module GradCam
"""
from typing import Any, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import Input
from tensorflow.python.keras import backend as K
from src.images.process_images import relu as relu_img
from numba import jit
from tqdm import tqdm
import cv2 as cv
import tensorflow_addons as tfa

def prob_grad_cam(
    cuts_images: tfa.types.TensorLike,
    classifier: List[str],
    last_conv_layer_name: str,
    paths_start_positions: tfa.types.TensorLike,
    model: Model,
    dim_orig: int = 1024,
    winner_pos: int = 0,
) -> tfa.types.TensorLike:
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
    dimensao_imagem = model.shape[1]
    dimensao_original = (dim_orig, dim_orig)
    # Inicializa a imagem final do grad cam com zeros
    grad_cam_prob = np.zeros(dimensao_original)
    # Armazena o numero de pacotes que passaram por um pixel
    splits_per_pixel = np.zeros(dimensao_original)
    # Recebe o tamanho da imagem a ser inserida no modelo
    grad_cam, resnet = modelo_grad_cam(
        model,
        last_conv_layer_name,
        dimensao_imagem
    )
    # Cria o modelo com as camadas após a ultima convolução
    model_after_last_conv = model_after(
        grad_cam,
        resnet,
        model,
        classifier
    )
    predicoes = model.predict(cuts_images)
    entrada_modelo = model.shape
    shape = (1, entrada_modelo[0], entrada_modelo[1], entrada_modelo[2])
    for predicao, cut_image, position_pixel in tqdm(
        zip(predicoes, cuts_images, paths_start_positions[0])
    ):
        # Os pacotes chegam com dimensões (None,224,224,3) e
        # precisam ser redimensionados para a (1,224,224,3)
        cut_image = cut_image.reshape(shape)
        # Define a predição do canal vencedor geral
        predicao_pacote = predicao[winner_pos]
        # Calcula o grad cam para o canal vencedor
        last_conv_layer_output = generate_grads_image(
            grad_cam,
            cut_image,
            model_after_last_conv
        )
        gradcam_pacote = create_heatmap(
            last_conv_layer_output,
            dimensao_imagem
        )
        # Multiplica a possibilidade do canal pelo grad cam gerado
        grad_cam_predicao = predicao_pacote * gradcam_pacote
        # Soma com a predicoes anteriores
        sum_grad_cam = sum_grads_cam(
            grad_cam_full=grad_cam_prob,
            grad_cam_cut=grad_cam_predicao,
            used_pixels=splits_per_pixel,
            start=position_pixel
        )
        grad_cam_prob, splits_per_pixel = sum_grad_cam
    # Divide o gradcam total pelas vezes que passou por um pixel
    div_grad_cam_prob = div_cuts_per_pixel(
        splits_per_pixel,
        grad_cam_prob
    )
    grad_cam_prob = normalize(div_grad_cam_prob)
    return grad_cam_prob

def modelo_grad_cam(
    modelo: Model,
    last_conv_layer_name: str,
    dim_image: int = 224
) -> Model:
    # Recebe o tamanho da imagem a ser inserida no modelo
    resnet = modelo.base
    # Nome da ultima camada de convolucao
    last_conv_layer = get_layer(resnet, last_conv_layer_name)
    # Cria um novo modelo com as camadas até a ultima convolução
    model_until_last_conv = Model(resnet.input, last_conv_layer.output)
    grad_cam_input = (dim_image, dim_image, 1)
    inputs = Input(shape=grad_cam_input)
    new_model = modelo.get_layer('conv_gray_rgb')(inputs)
    new_model = model_until_last_conv(new_model)
    new_model = Model(inputs, new_model)
    return new_model, resnet

def get_layer(resnet: Model, last_conv_layer_name: str) -> Layer:
    for layer in reversed(resnet.layers):
        if layer.name == last_conv_layer_name:
            return layer


# @jit(nopython=True)
def sum_grads_cam(
    grad_cam_full: tfa.types.TensorLike,
    grad_cam_cut: tfa.types.TensorLike,
    used_pixels: tfa.types.TensorLike,
    start: Tuple[int,int] = (0, 0),
) -> tfa.types.TensorLike:
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
    # Get the dimension of cut grad_cam
    dimension_cut = grad_cam_cut.shape[0]
    # Get original dimension
    dim_orig = grad_cam_full.shape[1]
    # Create the matriz of ones to sum to grad cam full
    ones = np.ones((dimension_cut, dimension_cut))
    final_x = int(start[0] + dimension_cut)
    final_y = int(start[1] + dimension_cut)
    
    final_x = final_x if final_x < dim_orig else dim_orig
    final_y = final_y if final_y < dim_orig else dim_orig
    
    grad_cam_full[int(start[0]):final_x, int(start[1]):final_y] += grad_cam_cut
    used_pixels[int(start[0]):final_x, int(start[1]):final_y] += ones
    return (grad_cam_full, used_pixels)


# @jit(nopython=True)
def div_cuts_per_pixel(
    splits_per_pixel: tfa.types.TensorLike,
    grad_cam_prob: tfa.types.TensorLike
) -> tfa.types.TensorLike:
    """ Divide os numeros de pacotes de pixeis pela
        grad cam do pacote gerado pela função para
        gerar o grad cam probabilistico.
        Args:
        -----
            pacotes_por_pixel (tfa.types.TensorLike):
                Numero dos pacotes passados em cada pixel.
            grad_cam_prob (tfa.types.TensorLike):
                GradCam do recorte da imagem.
        Returns:
        --------
            (np.array): GradCam Probabilistico
    """
    # Percorre a matrix
    for pixel_y, _ in enumerate(grad_cam_prob):
        for pixel_x, _ in enumerate(grad_cam_prob[0]):
            spltis = splits_per_pixel[pixel_y][pixel_x]
            if spltis > 0:
                # Divide cada pixel pelo numero de vezes de pacotes que utilizou esse pixel
                grad_cam_prob[pixel_y][pixel_x] /= spltis
    return grad_cam_prob


def find_base_model(model: Model) -> Model:
    for layer in model.layers:
        if type(model) == type(layer):
            return layer


def grad_cam(
    image: tfa.types.TensorLike,
    model: Model,
    classifier_layer_names: List[str],
    last_conv_layer_name: str = "avg_pool",
) -> tfa.types.TensorLike:
    """ Gera o mapa de processamento da CNN.
        Args:
            image (np.array): Recorte da imagem
            model (keras.Model): Modelo da CNN
        Returns:
            (np.array): Grad Cam
    """
    dim_image = model.input_shape[1]
    # Altera o dimensão da imagem para atender a entrada do modelo
    dimensao_modelo = (1, dim_image, dim_image, model.input_shape[3])
    image_reshape = image.reshape(dimensao_modelo)
    # Recebe o tamanho da imagem a ser inserida no modelo
    resnet = find_base_model(model)
    if resnet is not None:
        # Nome da ultima camada de convolucao
        last_conv_layer = resnet.get_layer(last_conv_layer_name)
        # Cria um novo modelo com as camadas até a ultima convolução
        model_until_last_conv = Model(resnet.input, last_conv_layer.output)
    else:
        # Nome da ultima camada de convolucao
        last_conv_layer = model.get_layer(last_conv_layer_name)
        # Cria um novo modelo com as camadas até a ultima convolucao
        model_until_last_conv = Model(model.input, last_conv_layer.output)
    
    inputs = Input(shape=(dim_image, dim_image, model.input_shape[3]))
    new_model = model.layers[1](inputs)
    new_model = model_until_last_conv(new_model)
    new_model = Model(inputs, new_model)
    
    # Cria o modelo com as camadas após a ultima convolução
    model_after_last_conv = model_after(
        new_model,
        resnet,
        model,
        classifier_layer_names
    )
    # Gera a 
    last_conv_layer_output = generate_grads_image(
        new_model,
        image_reshape,
        model_after_last_conv
    )
    heatmap = create_heatmap(last_conv_layer_output, dim_image)
    return heatmap


def generate_grads_image(
    model_until_last_conv: Model,
    image: tfa.types.TensorLike,
    classifier_model: Model
) -> tfa.types.TensorLike:
    """
        Compute the gradient of the top predicted
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
    tape: tf.GradientTape,
    last_cnn_output: Model,
    top_class_channel: List[List[int]]
) -> tfa.types.TensorLike:
    """"""
    # Calcula o gradiente do modelo para saída máxima (pesos)
    grads = tape.gradient(top_class_channel, last_cnn_output)
    # Retira pela media dos pixel (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_cnn_output[0]
    # Multiplicação matricias entre os dois elementos
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    return tf.squeeze(heatmap)

def model_after(
    last_conv_layer: Layer,
    resnet_model: Model,
    model: Model,
    classifier_layer_names: List[str]
) -> Model:
    """
        Cria apenas o modelo de classificação do modelo passado pelo usuario
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

def resize(image: tfa.types.TensorLike, dim: int) -> tfa.types.TensorLike:
    image = cv.resize(image, (dim, dim))
    return image

def create_heatmap(
    last_conv_layer_output: Layer,
    dim_split: int = 224
) -> tfa.types.TensorLike:
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
    # Normalização do heatmap
    heatmap = np.array(last_conv_layer_output)
    heatmap = relu_img(heatmap)
    heatmap = resize(heatmap, dim_split)
    return heatmap


def normalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == 0 and x_min == 0:
        return x
    x_new = (x - x_min) / (x_max - x_min)
    return x_new