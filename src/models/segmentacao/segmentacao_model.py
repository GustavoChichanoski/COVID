from typing import Any, List, Tuple
from tensorflow.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D, UpSampling2D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.regularizers import l1_l2

class Unet(Model):

    def __init__(
        self,
        dim: int = 256,
        activation: str = 'relu',
        n_class: int = 1,
        channels: int = 1,
        final_activation: str = 'sigmoid',
        filter_root: int = 32,
        name: str = 'UNet',
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

        # Propriedades da classe
        self._lazy_callbacks = None

    @property
    def callbacks(self):
        if self._lazy_callbacks is None:
            checkpoint = ModelCheckpoint(
                './.model/best.weights.hdf5', monitor='val_loss',
                verbose=1,save_best_only=True, mode='min',
                save_weights_only=True,
            )
            # Metrica para a redução do valor de LR
            reduceLROnPlat = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5,
                verbose=1, mode='min', epsilon=1e-2, cooldown=3,
                min_lr=1e-8
            )
            # Metrica para a parada do treino
            early = EarlyStopping(
                monitor='val_loss', mode='min',
                restore_best_weights=True, patience=40
            )
            # Vetor a ser passado na função fit
            self._lazy_callbacks = [checkpoint, early, reduceLROnPlat]
        return self._lazy_callbacks

    def Up_plus_Concatenate(
        self,
        layer: Layer,
        connection: Layer,
        i: int = 0
    ) -> Layer:
        # Define names of layers
        up_name = f'UpSampling{i}_1'
        conc_name = f'UpConcatenate{i}_1'
        # Create the layer sequencial
        layer = UpSampling2D(name=up_name)(layer)
        layer = Concatenate(
            axis=-1,name=conc_name
        )([layer, connection])
        return layer

    def conv_block(
        self,
        layer: Layer,
        act: str = 'relu',
        filters: int = 32,
        kernel: int = 3,
        i: int = 1,
        j: int = 1
    ) -> Layer:
        # Define os nomes das layers
        pos = f'{i}_{j}'
        conv_name = f'Conv{pos}'
        bn_name = f'BN{pos}'
        act_name = f'Act{pos}'

        layer = Conv2D(
            filters=filters,
            kernel_size=kernel,
            padding='same',
            kernel_regularizer=l1_l2(),
            name=conv_name
        )(layer)
        layer = BatchNormalization(name=bn_name)(layer)
        layer = Activation(act, name=act_name)(layer)
        return layer
    
    def call(self):

        store_layers = {}
        input_size = (self.dim, self.dim, self.channels)

        inputs = Input(input_size)

        first_layer = inputs

        for i in range(self.depth):

            filters = (2**i) * self.filter_root

            # Cria as duas convoluções da camada
            for j in range(2):
                layer = self.conv_block(
                    first_layer, filters, (3, 3),
                    self.activation, i, j
                )

            # Verifica se está na ultima camada
            if i < self.depth - 1:
                # Armazena a layer no dicionario
                store_layers[str(i)] = layer
                max_name = f'MaxPooling{i}_1'
                first_layer = MaxPooling2D(
                    (2, 2), padding='same', name=max_name
                )(layer)

            else:
                first_layer = layer

        for i in range(self.depth-2, -1, -1):

            filters = (2**i) * self.filter_root
            connection = store_layers[str(i)]

            layer = self.Up_plus_Concatenate(
                first_layer, connection, i
            )

            for j in range(2, 4):
                layer = self.conv_unet(
                    layer, filters, (3, 3),
                    self.activation, i, j
                )

            first_layer = layer
        return layer

    def fit(
        self, x, y,
        callbacks: List[Callback],
        validation_data: Tuple[Any,Any],
        batch_size: int = 32,
        epochs: int = 100,
        verbose: bool = True,
        validation_split: float = 0.3,
        shuffle: bool = True
    ):
        callbacks = self.callbacks if callbacks is None else callbacks
        return super().fit(
            x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
            callbacks=callbacks, validation_split=validation_split, 
            validation_data=validation_data, shuffle=shuffle
        )