import tensorflow as tf

from gc import garbage
from typing import Any, List, Optional, Tuple

from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Conv2D, UpSampling2D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizer_v2.adamax import Adamax
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.metrics import BinaryAccuracy
from tensorflow.python.keras.models import Input
from tensorflow.python.keras.utils.data_utils import Sequence

from src.models.losses.log_cosh_dice_loss import LogCoshDiceError
from src.models.metrics.f1_score import F1score
from src.models.losses.dice_loss import DiceError
from src.models.callbacks.clear_garbage import ClearGarbage

def inner_callbacks() -> List[Callback]:
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
    return [checkpoint, early, reduce_lr, terminate]


def inner_metrics() -> List[Metric]:
    return [
        BinaryAccuracy(name='bin_acc'),
        F1score(name='f1')
    ]

def unet_functional(
    inputs: Tuple[int,int,int],
    filter_root: int = 32,
    depth: int = 5,
    activation: str = 'relu',
    final_activation: str = 'sigmoid',
    n_class: int = 1,
    rate: float = 0.3
) -> Model:

    store_layers = {}
    first_layer = inputs
    k = 0
    input_layer = Input(inputs)
    first_layer = input_layer
    for i in range(depth):
        filters = (2 ** i) * filter_root
        # Cria as duas convoluções da camada
        layer = unet_conv(
            layer=first_layer,
            activation=activation,
            rate=rate,
            k=k,
            filters=filters
        )
        k += 1
        layer = unet_conv(
            layer=layer,
            activation=activation,
            rate=rate,
            k=k,
            filters=filters
        )
        k += 1
        # Verifica se está na ultima camada
        if i < depth - 1:
            # Armazena a layer no dicionario
            store_layers[str(i)] = layer
            first_layer = MaxPooling2D(
                pool_size=(2,2),
                padding='same',
                name=f'max_{k}'
            )(layer)
        else:
            first_layer = layer
    for i in range(depth-2,-1,-1):
        connection = store_layers[str(i)]
        filters = (2 ** i) * filter_root
        layer = UpSampling2D(name=f'up_{k}')(first_layer)
        layer = Concatenate(axis=-1, name=f'cat_{k}')([layer, connection])
        layer = unet_conv(
            layer=layer,
            activation=activation,
            rate=rate,
            k=k,
            filters=filters
        )
        k += 1
        layer = unet_conv(
            layer=layer,
            activation=activation,
            rate=rate,
            k=k,
            filters=filters
        )
        k += 1
        first_layer = layer
    layer = Dropout(rate=rate,name='dropout')(layer)
    outputs = Conv2D(
        filters=n_class,
        kernel_size=(1,1),
        padding='same',
        activation=final_activation,
        name='output'
    )(layer)
    model = Model(inputs=input_layer, outputs=outputs)
    return model

def unet_conv(
    layer: Layer,
    activation: str = 'relu',
    rate: float = 0.2,
    filters: int = 32,
    k: int = 0
) -> Layer:
    layer = Conv2D(
        filters=filters,
        kernel_size=(3,3),
        padding='same',
        name=f'conv_{k}'
    )(layer)
    layer = BatchNormalization(
        name=f'bn_{k}'
    )(layer)
    layer = Activation(
        activation,
        name=f'act_{k}'
    )(layer)
    layer = Dropout(
        rate=rate * 2, 
        name=f'drop_{k}'
    )(layer)
    return layer

def unet_fit(
    model: Model,
    x: Sequence,
    validation_data: Sequence,
    callbacks: Optional[List[Callback]] = None,
    batch_size: Optional[int] = None,
    epochs: int = 100,
    shuffle: bool = True,
    **params
) -> History:
    callbacks = inner_callbacks() if callbacks is None else callbacks
    return model.fit(
        x=x,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=shuffle,
        **params
    )

def unet_compile(
    model: Model,
    optimizer: str = 'adamax',
    loss: str = 'log_cosh_dice',
    metrics: Optional[List[Metric]] = None,
    lr: float = 1e-5,
    rf: float = 1.0,
    **params
) -> None:
    metrics = inner_metrics() if metrics is None else metrics
    optimizer = Adamax(learning_rate=lr) if optimizer == 'adamax' else optimizer
    loss_function = DiceError(regularization_factor=rf) if loss == 'dice' else loss
    loss_function = LogCoshDiceError(regularization_factor=rf) if loss == 'log_cosh_dice' else loss_function
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
        **params
    )
    return None

def save_weights(
    model: Model,
    filepath: str,
    overwrite: bool = True,
    **params
) -> None:
    model.save_weights(filepath, overwrite=overwrite, **params)
    return None