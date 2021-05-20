from segmentation_lung import model_unet
from typing import Tuple
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Layer
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from tensorflow.keras import activations


class UNet(Model):

    def __init__(
        self,
        input_shape: Tuple[int, int],
        depth: int = 5,
        n_class: int = 1,
        activation_layer: str = 'relu',
        activation_final_layer: str = 'sigmoid',
        filter_root: int = 32
    ) -> None:
        super(UNet, self).__init__()
        self.input_shape = input_shape
        self.depth = depth
        self.n_class = n_class
        self.activation_layer = activation_layer
        self.activation_final_layer = activation_final_layer
        self.filter_root = filter_root

    @property
    def model(self):
        return self.unet(self)
    
    def compile(self, lr: float = 1e-3):
        opt = Adam(lr=lr, loss=self.dice_coef_loss)
        self.model.compile()

    def unet(self):

        kernel_size = (3, 3)
        store_layers = {}
        inputs = Input(self.input_size)
        first_layer = inputs

        conv_params = {'kernel_size': kernel_size,
                       'activation': self.activation_layer}

        for i in range(self.depth):

            filters = (2**i) * self.filter_root

            # Cria as duas convoluções da camada
            for j in range(2):
                layer = self.conv_unet(
                    layer=first_layer, filters=filters, i=i, j=j, **conv_params)

            # Verifica se está na ultima camada
            if i < self.depth - 1:
                # Armazena a layer no dicionario
                store_layers[str(i)] = layer
                max_name = f'MaxPooling{i}_1'
                first_layer = MaxPooling2D(
                    (2, 2), padding='same', name=max_name)(layer)

            else:
                first_layer = layer

        for i in range(self.depth-2, -1, -1):

            filters = (2**i) * self.filter_root
            connection = store_layers[str(i)]

            layer = self.Up_plus_Concatenate(
                layer=first_layer, connection=connection, i=i)

            for j in range(2, 4):
                layer = self.conv_unet(
                    layer=layer, filters=filters, kernal_size=kernel_size, activation=self.activation, i=i, j=j)

            first_layer = layer

        layer = Dropout(0.33, name='Drop_1')(layer)
        outputs = Conv2D(filters=self.n_class, kernel_size=(1, 1), padding='same',
                         activation=self.final_activation, name='output')(layer)

        return Model(inputs, outputs, name="UNet")

    def conv_unet(
        layer: Layer,
        filters: int = 32,
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = 'relu',
        i: int = 0,
        j: int = 0
    ):
        # Define os nomes das layers
        conv_name = f"Conv{i}_{j}"
        act_name = f"Act{i}_{j}"

        layer = Conv2D(filters=filters, kernel_size=kernel_size,
                       padding='same', name=conv_name)(layer)
        layer = Activation(activation, name=act_name)(layer)
        return layer

    def Up_plus_Concatenate(layer: Layer, connection: Layer, i: int = 0):
        # Define names of layers
        up_name = f'UpSampling{i}_1'
        conc_name = f'UpConcatenate{i}_1'
        # Create the layer sequencial
        layer = UpSampling2D(name=up_name)(layer)
        layer = Concatenate(axis=-1, name=conc_name)([layer, connection])
        return layer

def dice_coef(y_true, y_pred):
    """Dice Coefficient
    Project: BraTs   Author: cv-lee   File: unet.py    License: MIT License
    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    Returns:
        (np.array): Calcula a porcentagem de acerto da rede neural
    """

    class_num = 1

    for class_now in range(class_num):

        # Converte y_pred e y_true em vetores
        y_true_f = K.flatten(y_true[:, :, :, class_now])
        y_pred_f = K.flatten(y_pred[:, :, :, class_now])

        # Calcula o numero de vezes que
        # y_true(positve) é igual y_pred(positive) (tp)
        intersection = K.sum(y_true_f * y_pred_f)
        # Soma o número de vezes que ambos foram positivos
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        # Smooth - Evita que o denominador fique muito pequeno
        smooth = K.constant(1e-6)
        # Calculo o erro entre eles
        num = (K.constant(2)*intersection + 1)
        den = (union + smooth)
        loss = num / den

        if class_now == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss

        total_loss = total_loss / class_num

    return total_loss

def dice_coef_loss(y_true, y_pred):
    accuracy = 1 - dice_coef(y_true, y_pred)
    return accuracy