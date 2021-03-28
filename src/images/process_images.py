"""
    Biblioteca referente ao processamento das imagens
"""
from src.images.read_image import read_images
import numpy as np
from tqdm import tqdm
import cv2 as cv
from src.plots.plots import plot_images

DIM_SPLIT = 224
DIM_ORIG = 1024
K_SPLIT = 100
SCALE = 255
THRESHOLD = 1279488 # 224 * 224 * 255 * .1

def invert_image(image):
    '''
        Função para inverter a imagem
        Args:
            Recebe uma imagem de np.array e inverte os valores
        return:
            Retorna a imagem como sendo um vetor np.array
    '''
    return cv.bitwise_not(image)


def equalize_histogram(image):
    """
        Equaliza o histograma da imagem

        Args:
            image (list): array com as valores da imagem

        Returns:
            list: retorna a imagem com o histograma equalizado
    """
    return cv.equalizeHist(image)


def resize_image(image, dim: int):
    image = cv.resize(image, (dim, dim))
    return image


def find_start(image_with_black) -> tuple:
    """
        Encontra o primeiro pixel não zero da esquerda para a direita.
        Args:
            image_with_black (np.array): Imagem a ser analizada.
            size (int, optional): Tamanho da imagem a ser analizada.
                                  Defaults to 1024.
        Returns:
            (tuple): Primeira linha e coluna contendo um pixel não zero.
    """
    size = image_with_black.shape[0]
    if isinstance(image_with_black, list):
        start = []
        for img in image_with_black:
            row, column = find_start(img, size)
            start.append((row, column))
        return start
    row_start, column_start = 0, 0
    # percorre a imagem da cima para baixo
    for i in range(size):
        if np.sum(image_with_black[i]) > 0:
            row_start = i
            break
    # percorre a imagem da esquerda para a direita
    for j in range(size):
        if np.sum(image_with_black[:, j]) > 0:
            column_start = j
            break
    return row_start, column_start


def find_end(image_with_black,
             size: int = 1024):
    """
        Encontra o primeiro pixel não zero da direita para a esquerda0

        Args:
            image_with_black (np.array): imagem a ser analizada
            size (int, optional): tamanho da imagem a ser analizada.
                                  Defaults to 1024.

        Returns:
            (tuple): Primeira linha e coluna contendo um pixel não zero.
    """
    if isinstance(image_with_black, list):
        ends = []
        for image in image_with_black:
            row, column = find_end(image, size)
            ends.append((row, column))
        return ends
    row_end, column_end = 0, 0
    # percorre a imagem da baixo para cima
    for i in range(size - 1, -1, -1):
        if np.sum(image_with_black[i]) > 0:
            row_end = i
            break
    # percorre a imagem da direita para a esquerda
    for j in range(size - 1, -1, -1):
        if np.sum(image_with_black[:, j]) > 0:
            column_end = j
            break
    return row_end, column_end


def random_pixel(start: tuple = (0, 0),
                 end: tuple = (0, 0),
                 dim_split: int = DIM_SPLIT):
    """
        Seleciona um pixel randomicamente comecando de start e
        indo end menos a dimensão maxima do corte.

        Args:
            start (tuple, optional): Pixel superior. 
                                     Defaults to (0,0).
            end (tuple, optional): Pixel inferior. 
                                   Defaults to (0,0).
            dim_split (int, optional): Dimensão do corte.
                                       Defaults to 224.

        Returns:
            (tuple): pixel gerados aleatoriamente
    """
    x_i, y_i = start
    x_e, y_e = end
    try:
        pixel_x = np.random.randint(x_i, x_e-dim_split)
        pixel_y = np.random.randint(y_i, y_e-dim_split)
    except:
        print(start)
        print(end)
    return pixel_x, pixel_y


def rescale_images(original_image, scale: int = 255):
    """
        Rescala a imagem para ir de -1 a 1

        Args:
            original_image (list or np.array): imagem ainda não rescalada
            scale (int, optional): escala da nova imagem. 
                                   Defaults to 255.

        Returns:
            (list or np.array) : imagem rescalada
    """
    if isinstance(original_image, list):
        rescales = []
        for img in original_image:
            scale_img = rescale_images(img, scale)
            rescales.append(scale_img)
        return rescales
    half_scale = scale/2
    return (original_image-half_scale)/half_scale


def normalize_image(images):
    """
        Normaliza as imagens para que todos variem de 0 a 1.

        Args:
            images (list or np.array): Pode ser uma lista de imagens ou uma imagem.

        Returns:
            (np.array): Imagens normalizadas
    """
    x_new = images / 255
    return x_new


def gray2rgb(gray_image):
    """
        Transforma imagens em escala de cinza em coloridas.

        Args:
            gray_image (np.array): Imagem em escala de cinza.

        Returns:
            (np.array): Imagens colorida.
    """
    if isinstance(gray_image, list):
        coloreds = []
        for gray in gray_image:
            colored = gray2rgb(gray)
            coloreds.append(colored)
        return coloreds
    return cv.cvtColor(gray_image, cv.COLOR_GRAY2RGB)


def bgr2gray(colored_images):
    """
        Transforma imagens coloridas em escala de cinza.

        Args:
            colored_images (np.array): Imagem colorida.

        Returns:
            (np.array): Imagens em escala cinza.
    """
    if isinstance(colored_images, list):
        grays = []
        for color in colored_images:
            gray = bgr2gray(color)
            grays.append(gray)
        return grays
    else:
        return cv.cvtColor(colored_images, cv.COLOR_BGR2GRAY)


def split_images_n_times(image,
                         n_split: int = 100,
                         dim_split: int = 224,
                         verbose: bool = True):
    """
        Recorta a imagem em n_split vezes de tamanhos dim_split ignorando
        recortes totalmente pretos.

        Args:
            image (np.array): imagem a ser recortada
            n_split (int, optional): Numero de cortes. Defaults to 100.
            dim_orig (int, optional): Tamanho da imagem. Defaults to 1024.
            dim_split (int, optional): Tamanho dos cortes. Defaults to 224.

        Returns:
            (tuple): recortes das imagens e o pixel inicial.
    """
    # Criação das listas
    cut_img = []  # lista de cortes
    cut_pos = []  # lista de posicoes do corte
    # Define os pixels em que a imgem começa
    pixel_start = find_start(image)
    pixel_end = find_end(image)
    # Cria os n_splits cortes
    pbar = range(n_split)
    if verbose:
        pbar = tqdm(pbar)
    for _ in pbar:
        # Recebe um corte da imagem não inteiramente preto
        cut, pos = create_non_black_cut(image,
                                        pixel_start, pixel_end,
                                        dim_split)
        cut_norm = normalize_image(cut)
        cut_img = np.append(cut_img, cut_norm)  # Armazena o corte
        cut_pos.append(pos)  # Armaxena o pixel inicial do corte
    return cut_img, cut_pos

def split_images(image, dim: int = 224):
    # Define os pixels em que a imgem começa
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    pixel_start, pixel_end = (np.min(y_nonzero),np.max(y_nonzero)), (np.min(x_nonzero),np.max(x_nonzero))

    # Recebe um corte da imagem não inteiramente preto
    cut, _ = create_non_black_cut(image,
                                  pixel_start,
                                  pixel_end,
                                  dim)
    cut_norm = normalize_image(cut)
    return cut_norm


def create_non_black_cut(image,
                         start: tuple = (0, 0),
                         end: tuple = (0, 0),
                         dim: int = 224):
    """
        Cria um recorte que não é totalmente preto

        Args:
            image (np.array): Imagem a ser cortada
            start (tuple, optional): Pixel por onde comecar a cortar.
                                    Defaults to (0, 0).
            end (tuple, optional): Pixel para parar de corte.
                                Defaults to (0, 0).
            dim (int, optional): Dimensão do corte. Defaults to 224.

        Returns:
            numpy.array: recorte da imagem nao totalmente preta
    """
    if start[1] > end[1] - dim:
        end = (end[0], start[1] + dim + 10)
        pos = random_pixel(start, end, dim)
        recort = create_recort(image, pos, dim)
        return recort, pos
    if start[0] > end[0] - dim:
        end = (start[0] + dim + 10, end[1])
        pos = random_pixel(start, end, dim)
        recort = create_recort(image, pos, dim)
        return recort, pos
    pos = random_pixel(start, end, dim)
    recort = create_recort(image, pos, dim)
    soma = np.sum(recort)
    while soma < THRESHOLD:
        pos = random_pixel(start, end, dim)
        recort = create_recort(image, pos, dim)
        soma = np.sum(recort)
    return recort, pos

def create_recort(image,
                  pos_start: tuple = (0, 0),
                  dim_split: int = 224):
    """
        Cria um recorte da imagem indo da posicao inicial até a 
        dimensão do recorte

        Args:
            image (np.array): Imagem a ser recortada.
            pos_start (tuple, optional): Posicao do recorte.
                                         Defaults to (0,0).
            dim_split (int, optional): Dimensão do recorte.
                                       Defaults to 224.

        Return:
            (np.array): Recorte da imagem
    """
    pos_end = (pos_start[0]+dim_split, pos_start[1]+dim_split)
    cut = image[pos_start[0]:pos_end[0], pos_start[1]:pos_end[1]]
    return cut


def relu(image):
    """
        Retifica a imagem
        Args:
        -----
            image: imagem a ser retificada. (np.array)
        Returns:
        -------
            (np.array): imagem retificada. (np.array)
    """
    return np.clip(image, 0, None)
