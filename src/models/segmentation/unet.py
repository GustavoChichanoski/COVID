from gc import garbage
from typing import Any, List, Optional, Tuple

from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.convolutional import Conv2D, UpSampling2D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.metrics import BinaryAccuracy
from tensorflow.python.keras.models import Input
from tensorflow.python.keras.utils.data_utils import Sequence

from src.data.generator import KerasGenerator
from src.models.losses.log_cosh_dice_loss import LogCoshDiceError
from src.models.metrics.f1_score import F1score
from src.models.losses.dice_loss import DiceError
from src.models.callbacks.clear_garbage import ClearGarbage
import tensorflow as tf

class Unet(Model):

    def __init__(
        self,
        dim: int = 256,
        activation: str = 'relu',
        n_class: int = 1,
        channels: int = 1,
        depth: int = 5,
        final_activation: str = 'sigmoid',
        filter_root: int = 32,
        name: str = 'UNet',
        rate: float = 0.33,
        **kwargs
    ) -> None:
        super().__init__(name=name,**kwargs)

        # Parametros
        self.dim = dim
        self.n_class = n_class
        self.activation = activation
        self.final_activation = final_activation
        self.channels = channels
        self.filter_root = filter_root
        self.depth = depth
        self.rate = rate if rate < 0.375 else 0.375

        # Repeateble layers
        self.conv = [Layer] * (self.depth * 4 - 2)
        self.bn = [Layer] * (self.depth * 4 - 2)
        self.act = [Layer] * (self.depth * 4 - 2)
        self.drop = [Layer] * (self.depth * 4 - 1)
        self.max = [Layer] * (self.depth - 1)
        self.up = [Layer] * (self.depth - 1)
        self.cat = [Layer] * (self.depth - 1)

        self.kernel = (3,3)

        k = 0

        for i in range(depth):
            filters = (2 ** i) * filter_root
            for _ in range(2):
                conv_name = f'conv_{k}'
                self.conv[k] = self._conv(filters,conv_name)
                k += 1
        for i in range(depth-2,-1,-1):
            filters = (2 ** i) * filter_root
            for _ in range(2):
                conv_name = f'conv_{k}'
                self.conv[k] = self._conv(filters, conv_name)
                k += 1

        self.bn = [
            BatchNormalization(name=f'bn_{k}')
            for k in range(len(self.bn))
        ]
        self.act = [
            Activation(self.activation, name=f'act_{k}')
            for k in range(len(self.act))
        ]
        self.drop = [
            Dropout(
                rate=self.rate * 2,
                name=f'drop_{k}'
            ) for k in range(len(self.drop))
        ]
        self.drop[-1] = Dropout(rate=self.rate,name='dropout')

        self.max = [
            MaxPooling2D((2,2), padding='same', name=f'max_{k}')
            for k in range(len(self.max))
        ]
        self.up = [
            UpSampling2D(name=f'up_{k}')
            for k in range(len(self.up))
        ]
        self.cat = [
            Concatenate(axis=-1,name=f'cat_{k}')
            for k in range(len(self.cat))
        ]

        # Unique layers
        self.last_conv = Conv2D(
            n_class,
            (1,1),
            padding='same',
            activation=final_activation,
            name='output'
        )

        # Propriedades da classe
        self._lazy_callbacks: Optional[List[Callback]] = None
        self._lazy_metrics: Optional[List[Metric]] = None

        # Input layer
        input_shape = (self.dim, self.dim, self.channels)
        self.input_layer = Input(input_shape)
        self.output_layer = self.call(self.input_layer)

    def _conv(self, filters: int = 32, conv_name: str = 'conv') -> Conv2D:
        return Conv2D(
            filters=filters,
            kernel_size=self.kernel,
            padding='same',
            name=conv_name
        )

    @property
    def inner_callbacks(self) -> List[Callback]:
        if self._lazy_callbacks is None:
            checkpoint = ModelCheckpoint(
                '.\\best.weights.hdf5', monitor='val_f1',
                verbose=1,save_best_only=True, mode='max',
                save_weights_only=True,
            )
            # Metrica para a redução do valor de LR
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3,
                verbose=1, mode='min', min_delta=1e-2, cooldown=3,
                min_lr=1e-8
            )
            # Metrica para a parada do treino
            early = EarlyStopping(
                monitor='val_loss', mode='min',
                restore_best_weights=True, patience=10
            )
            terminate = TerminateOnNaN()
            # Limpar o lixo do python
            garbage = ClearGarbage()
            # Vetor a ser passado na função fit
            self._lazy_callbacks = [checkpoint, early, reduce_lr, terminate]
        return self._lazy_callbacks

    @property
    def inner_metrics(self) -> List[Metric]:
        if self._lazy_metrics is None:
            self._lazy_metrics = [
                BinaryAccuracy(name='bin_acc'),
                F1score(name='f1')
            ]
        return self._lazy_metrics

    def call(
        self,
        inputs: Tuple[int,int,int]
    ) -> Model:

        store_layers = {}
        first_layer = inputs
        k = 0
        for i in range(self.depth):
            # Cria as duas convoluções da camada
            layer = self.unet_conv(first_layer, k)
            k += 1
            layer = self.unet_conv(layer, k)
            k += 1
            # Verifica se está na ultima camada
            if i < self.depth - 1:
                # Armazena a layer no dicionario
                store_layers[str(i)] = layer
                first_layer = self.max[i](layer)
            else:
                first_layer = layer
        for i in range(self.depth-2,-1,-1):
            connection = store_layers[str(i)]
            layer = self.up[i](first_layer)
            # self.cat[i].build(shape)
            layer = self.cat[i]([layer, connection])
            layer = self.unet_conv(layer, k)
            k += 1
            layer = self.unet_conv(layer, k)
            k += 1
            first_layer = layer
        layer = self.drop[-1](layer)
        return self.last_conv(layer)

    def unet_conv(
        self,
        layer: Layer,
        k: int
    ) -> Layer:
        layer = self.conv[k](layer)
        layer = self.act[k](layer)
        layer = self.bn[k](layer)
        layer = self.drop[k](layer)
        return layer

    def fit(
        self,
        x: Sequence,
        validation_data: Sequence,
        callbacks: Optional[List[Callback]] = None,
        batch_size: Optional[int] = None,
        epochs: int = 100,
        shuffle: bool = True,
        **params
    ) -> Any:
        callbacks = self.inner_callbacks if callbacks is None else callbacks
        return super().fit(
            x=x,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=shuffle,
            **params
        )

    def build(self) -> None:
        is_tensor = tf.is_tensor(self.input_layer)
        super(Unet, self).build(
            self.input_layer.shape if is_tensor else self.inputs_layer
        )
        self.call(self.input_layer)

    def compile(
        self,
        optimizer: str = 'adamax',
        loss: str = 'log_cosh_dice',
        metrics: Optional[List[Metric]] = None,
        lr: float = 1e-5,
        rf: float = 1.0,
        **params
    ) -> None:
        metrics = self.inner_metrics if metrics is None else metrics
        optimizer = Adamax(learning_rate=lr) if optimizer == 'adamax' else optimizer
        loss_function = DiceError(regularization_factor=rf) if loss == 'dice' else loss
        loss_function = LogCoshDiceError(regularization_factor=rf) if loss == 'log_cosh_dice' else loss_function
        super().compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics,
             **params
        )
        return None

    def save_weights(
        self,
        filepath: str,
        overwrite: bool = True,
        **params
    ) -> None:
        super().save_weights(filepath, overwrite=overwrite, **params)

    def load_weights(self, filepath: str, **params) -> None:
        super().load_weights(filepath, **params)

    def predict(self, x: KerasGenerator, **params) -> Any:
        return super().predict(x, **params)
