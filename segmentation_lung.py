import os
import cv2 as cv
import numpy as np  # linear algebra
import sys
import random as rd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import LocallyConnected2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import TruePositives
from keras.metrics import TrueNegatives
from keras.metrics import FalsePositives
from keras.metrics import FalseNegatives
from keras.metrics import BinaryAccuracy
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
%matplotlib inline


sys.path.append('./Covid')
from src.model.metrics.f1_score import F1score
from src.dataset.dataset_seg import SegmentationDataset
from src.dataset.dataset_seg import SegmentationDataGenerator

def conv_unet(
    layer,
    filters=32,kernel=(3,3),
    act="relu",
    i=1,j=1):
    # Define os nomes das layers
    conv_name = f"Conv{i}_{j}"
    bn_name = f"BN{i}_{j}"
    act_name = f"Act{i}_{j}"
    drop_name = f"Drop{i}_{j}"
    layer = Conv2D(
        filters=filters,kernel_size=kernel,
        padding='same', name=conv_name)(layer)
    layer = BatchNormalization(name=bn_name)(layer)
    layer = Activation(act,name=act_name)(layer)
    layer = Dropout(0.75,name=drop_name)(layer)
    return layer
    
def Up_plus_Concatenate(layer,connection,i):
    # Define names of layers
    up_name = 'UpSampling{}_1'.format(i) 
    conc_name = 'UpConcatenate{}_1'.format(i)
    # Create the layer sequencial
    layer = UpSampling2D(name=up_name)(layer)
    layer = Concatenate(axis=-1,
        name=conc_name)([layer,connection])
    return layer

def model_unet(
    input_size=(None,256,256,1),
    depth=5,
    activation='relu',
    n_class=1,
    final_activation='sigmoid',
    filter_root=32):
    
    store_layers = {}
    
    inputs = Input(input_size)
    
    first_layer = inputs
    
    for i in range(depth):
    
        filters = (2**i) * filter_root

        # Cria as duas convoluções da camada
        for j in range(2):
            layer = conv_unet(
                first_layer,filters,(3,3),
                activation,i,j)

        # Verifica se está na ultima camada
        if i < depth - 1:
            # Armazena a layer no dicionario
            store_layers[str(i)] = layer
            max_name = 'MaxPooling{}_1'.format(i)
            first_layer = MaxPooling2D(
                (2,2),padding='same',
                name=max_name
            )(layer)

        else:
            first_layer = layer

    for i in range(depth-2,-1,-1):

        filters = (2**i) * filter_root
        connection = store_layers[str(i)]

        layer = Up_plus_Concatenate(first_layer,connection,i)

        for j in range(2,4):
            layer = conv_unet(
                layer,filters,(3,3),
                activation,i,j)

        first_layer = layer
        
    layer = Dropout(0.5,name='Drop_1')(layer)
    outputs = Conv2D(
        n_class,(1,1),padding='same',
        activation=final_activation, name='output'
    )(layer)
    
    return Model(inputs,outputs,name="UNet")

def dice_coef(y_true, y_pred):
    ''' Dice Coefficient
    Project: BraTs   Author: cv-lee   File: unet.py    License: MIT License
    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    Returns:
        (np.array): Calcula a porcentagem de acerto da rede neural
    '''

    class_num = 1

    for class_now in range(class_num):
    
    # Converte y_pred e y_true em vetores
        y_true_f = K.flatten(y_true[:,:,:,class_now])
        y_pred_f = K.flatten(y_pred[:,:,:,class_now])

        # Calcula o numero de vezes que
        # y_true(positve) é igual y_pred(positive) (tp)
        intersection = K.sum(y_true_f * y_pred_f)
        # Soma o número de vezes que ambos foram positivos
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        # Smooth - Evita que o denominador fique muito pequeno
        smooth = K.constant(1e-6);
        # Calculo o erro entre eles
        num = (K.constant(2) * intersection + 1)
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

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2021-06-11T14:38:13.889895Z","iopub.execute_input":"2021-06-11T14:38:13.890274Z","iopub.status.idle":"2021-06-11T14:38:13.900821Z","shell.execute_reply.started":"2021-06-11T14:38:13.890238Z","shell.execute_reply":"2021-06-11T14:38:13.900169Z"}}
def fill_subplot(axs,x,y1,y2,legend,title):
    axs.plot(x,y1,'r--')
    axs.plot(x,y2,'b--')
    axs.legend(legend)
    axs.set_title(title)
    return axs


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:38:13.988560Z","iopub.execute_input":"2021-06-11T14:38:13.988837Z","iopub.status.idle":"2021-06-11T14:38:13.993008Z","shell.execute_reply.started":"2021-06-11T14:38:13.988812Z","shell.execute_reply":"2021-06-11T14:38:13.992167Z"}}
DIM = 512
EPOCH = 300
IMG_SIZE = (DIM,DIM)
PATH = Path('../input/chest-xray-masks-and-labels/Lung Segmentation')
PATH_MODEL = 'model.h5'
BATCH_SIZE = 16

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:38:14.079547Z","iopub.execute_input":"2021-06-11T14:38:14.079786Z","iopub.status.idle":"2021-06-11T14:38:14.084451Z","shell.execute_reply.started":"2021-06-11T14:38:14.079763Z","shell.execute_reply":"2021-06-11T14:38:14.083570Z"}}
lung_path = PATH / 'CXR_png'
mask_path = PATH / 'masks'
test_path = PATH / 'test'
weight_path = "cxr_reg_weights.best.hdf5"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:38:14.191160Z","iopub.execute_input":"2021-06-11T14:38:14.191424Z","iopub.status.idle":"2021-06-11T14:41:18.293512Z","shell.execute_reply.started":"2021-06-11T14:38:14.191399Z","shell.execute_reply":"2021-06-11T14:41:18.292421Z"}}
# Fixa a aleatoridade do numpy
np.random.seed(42)
ds_train = SegmentationDataset(path_lung=lung_path, path_mask=mask_path)
ds_test = SegmentationDataset(path_lung=test_path)

part_param = {"tamanho": 0}
train, validation = ds_train.partition(val_size=0.2, **part_param)
test_values, _test_val_v = ds_test.partition(val_size=1e-5, **part_param)

params = {
    "dim": DIM_SPLIT,
    "batch_size": BATCH_SIZE
}
train_generator = SegmentationDataGenerator(x_set=train[0], y_set=train[1], **params)
val_generator = SegmentationDataGenerator(x_set=validation[0], y_set=validation[1], **params)
test_generator = SegmentationDataGenerator(x_set=test_values[0], y_set=test_values[1], **params)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:41:24.315437Z","iopub.execute_input":"2021-06-11T14:41:24.315839Z","iopub.status.idle":"2021-06-11T14:41:24.394821Z","shell.execute_reply.started":"2021-06-11T14:41:24.315788Z","shell.execute_reply":"2021-06-11T14:41:24.393909Z"}}
m = F1score()
metrics = [
#     TruePositives(name='tp'),  # Valores realmente positivos
#     TrueNegatives(name='tn'),  # Valores realmente negativos
#     FalsePositives(name='fp'), # Valores erroneamente positivos
#     FalseNegatives(name='fn'), # Valores erroneamente negativos
    BinaryAccuracy(name='accuracy'),
    m # F1Score
]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:41:24.399707Z","iopub.execute_input":"2021-06-11T14:41:24.400003Z","iopub.status.idle":"2021-06-11T14:41:24.413670Z","shell.execute_reply.started":"2021-06-11T14:41:24.399965Z","shell.execute_reply":"2021-06-11T14:41:24.412956Z"}}
# Metrica de salvamento
checkpoint = ModelCheckpoint(
    weight_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=True
)
# Metrica para a redução do valor de LR
reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=5,
    verbose=1,
    mode='min',
    epsilon=1e-2,
    cooldown=4,
    min_lr=1e-8
)
# Metrica para a parada do treino
early = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=10
)
# Vetor a ser passado na função fit
callbacks_list = [checkpoint, reduceLROnPlat, early]

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:24.416826Z","iopub.execute_input":"2021-06-11T14:41:24.417390Z","iopub.status.idle":"2021-06-11T14:41:25.160895Z","shell.execute_reply.started":"2021-06-11T14:41:24.417349Z","shell.execute_reply":"2021-06-11T14:41:25.157724Z"}}
# Criação e compilação do modelo 1 proposto
filters = 32
depth = 5
model = model_unet(
    (DIM,DIM,1),
    filter_root=filters,
    depth=depth,
    activation='relu'
)
# model.compile(
#     optimizer=Adam(lr=1e-5),
#     loss='binary_crossentropy',
#     metrics=metrics
# )
model.compile(
    optimizer=Adam(lr=1e-5),
    loss=dice_coef_loss,
    metrics=metrics
)
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:25.162323Z","iopub.execute_input":"2021-06-11T14:41:25.162660Z","iopub.status.idle":"2021-06-11T14:41:25.171231Z","shell.execute_reply.started":"2021-06-11T14:41:25.162623Z","shell.execute_reply":"2021-06-11T14:41:25.167942Z"}}
from keras.utils import plot_model
# plot_model(model, "my_first_model.png",show_shapes=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:41:25.173043Z","iopub.execute_input":"2021-06-11T14:41:25.173446Z","iopub.status.idle":"2021-06-11T14:41:25.178724Z","shell.execute_reply.started":"2021-06-11T14:41:25.173407Z","shell.execute_reply":"2021-06-11T14:41:25.177360Z"}}
history = []

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:25.180950Z","iopub.execute_input":"2021-06-11T14:41:25.181488Z","iopub.status.idle":"2021-06-11T14:41:49.665636Z","shell.execute_reply.started":"2021-06-11T14:41:25.181450Z","shell.execute_reply":"2021-06-11T14:41:49.663242Z"}}
history.append(
    model.fit(
        x=train_generator,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        validation_data=val_generator,
        callbacks=callbacks_list,
        workers=1
    )
)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:41:49.666926Z","iopub.status.idle":"2021-06-11T14:41:49.667854Z"}}
training_loss, validation_loss = [], []
training_accuracy, validation_accuracy = [], []

training_tp, validation_tp = [], []
training_tn, validation_tn = [], []
training_fp, validation_fp = [], []
training_fn, validation_fn = [], []

training_f1, validation_f1 = [], []

total = (EPOCH*DIM*DIM)

for i in history:

    training_loss += i.history['loss']
    validation_loss += i.history['val_loss']
    
    training_accuracy += i.history['accuracy']
    validation_accuracy += i.history['val_accuracy']
    
#     training_tp += i.history['tp']
#     validation_tp += i.history['val_tp']
    
#     training_fp += i.history['fp']
#     validation_fp += i.history['val_fp']
    
#     training_tn += i.history['tn']
#     validation_tn += i.history['val_tn']
    
#     training_fn += i.history['fn']
#     validation_fn += i.history['val_fn']
    
    training_f1 += i.history['f1']
    validation_f1 += i.history['val_f1']

# %% [code] {"jupyter":{"outputs_hidden":false},"scrolled":true,"execution":{"iopub.status.busy":"2021-06-11T14:41:49.669430Z","iopub.status.idle":"2021-06-11T14:41:49.670466Z"}}
val_acc = (1 - validation_loss[-1])*100
model.save(
    'model_{}_{}_acc_{:.02f}.h5'.format(depth,filters,val_acc),
    overwrite=True
)
model.save_weights(
    'weight_val_acc_{:.02f}.h5'.format(val_acc),
    overwrite=False
)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.672290Z","iopub.status.idle":"2021-06-11T14:41:49.673018Z"}}
'weight_val_acc_{:.02f}.h5'.format(val_acc)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.674546Z","iopub.status.idle":"2021-06-11T14:41:49.675257Z"}}
fig, axs = plt.subplots(1,2,figsize=(10,10))
    
epoch_count = range(1,len(training_loss) + 1)

str_val, str_train = "Validação ", "Treinamento "
str_acc, str_loss = "Acuracia ", "Erro "
str_p, str_n = "Positives ", "Negatives "
str_f, str_t = "False ", "True "
str_f1score = "F1Score"

str_train_loss = f"{str_train}{str_loss}"
str_val_loss = f"{str_val}{str_loss}"

str_train_acc = f"{str_train}{str_acc}"
str_val_acc = f"{str_val}{str_acc}"

str_fp = f"{str_f}{str_p}"
str_fn = f"{str_f}{str_n}"

str_tp = f"{str_t}{str_p}"
str_tn = f"{str_t}{str_n}"

str_train_fn = f"{str_train}{str_fn}"
str_train_fp = f"{str_train}{str_fp}"

str_val_fn = f"{str_val}{str_fn}"
str_val_fp = f"{str_val}{str_fp}"

str_train_tn = f"{str_train}{str_tn}"
str_train_tp = f"{str_train}{str_tp}"

str_val_tn = f"{str_val}{str_tn}"
str_val_tp = f"{str_val}{str_tp}"

str_train_f1 = f'{str_train}{str_f1score}'
str_val_f1 = f'{str_val}{str_f1score}'

axs[0] = fill_subplot(axs=axs[0],
    x=epoch_count, title=str_loss,
    y1=training_loss,y2=validation_loss,
    legend=[str_train_loss,str_val_loss])
axs[1] = fill_subplot(axs=axs[1],
    x=epoch_count,title=str_acc,
    y1=training_accuracy,y2=validation_accuracy,
    legend=[str_train_acc,str_val_acc])
# axs[1,0] = fill_subplot(axs=axs[1,0],
#     x=epoch_count,
#     y1=training_tp,y2=training_fp,
#     legend=[str_train_fp,str_train_tp],
#     title='Trainig Positives: True vs False')
# axs[1,1] = fill_subplot(axs=axs[1,1],
#     x=epoch_count,
#     y1=validation_tp,y2=validation_fp,
#     legend=[str_val_tp,str_val_fp],
#     title='Validation Positives: True vs False')
# axs[2,0] = fill_subplot(axs=axs[2,0],
#     x=epoch_count,
#     y1=training_tn,y2=training_fn,
#     legend=[str_train_tn,str_train_fn],
#     title='Trainig Negatives: True vs False')
# axs[2,1] = fill_subplot(axs=axs[2,1],
#     x=epoch_count,
#     y1=validation_tn,y2=validation_fn,
#     legend=[str_val_tn,str_val_fn],
#     title='Validation Negatives: True vs False')

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.676918Z","iopub.status.idle":"2021-06-11T14:41:49.677930Z"}}
from matplotlib.pyplot import figure
figure(figsize=(8,6),dpi=80)
plt.plot(epoch_count,training_loss,'r--')
plt.plot(epoch_count,validation_loss,'b--')
plt.legend([str_train_loss,str_val_loss],fontsize=14)
plt.xlabel('Épocas',fontsize=14)
plt.ylabel('Erro',fontsize=14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title('Erro ao longo das épocas [treino: {:.2f}% val: {:.2f}%]'.format(training_loss[-1]*100,validation_loss[-1]*100),fontsize=16)
plt.grid()
name_erro = './erro.png'
plt.savefig(name_erro)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.679444Z","iopub.status.idle":"2021-06-11T14:41:49.680358Z"}}
figure(figsize=(8,6),dpi=80)
plt.plot(epoch_count,training_f1,'r--')
plt.plot(epoch_count,validation_f1,'b--')
plt.legend([str_train_acc,str_val_acc],fontsize=14)
plt.xlabel('Épocas',fontsize=14)
plt.ylabel('F1',fontsize=14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title('F1 ao longo das épocas [treino: {:.2f}% val: {:.2f}%]'.format(training_f1[-1]*100,validation_f1[-1]*100),fontsize=16)
plt.grid()
name_erro = './erro.png'
plt.savefig(name_erro)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.681841Z","iopub.status.idle":"2021-06-11T14:41:49.682729Z"}}
figure(figsize=(8,6),dpi=80)
plt.plot(epoch_count,training_accuracy,'r--')
plt.plot(epoch_count,validation_accuracy,'b--')
plt.legend(['Treinamento','Validação'],fontsize=14)
plt.xlabel('Épocas',fontsize=14)
plt.ylabel('Acurácia',fontsize=14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title('Acurácia ao longo das épocas \n [treino: {:.2f}% val: {:.2f}%]'.format(training_accuracy[-1] * 100,validation_accuracy[-1] * 100),fontsize=16)
plt.grid()
name_erro = './erro.png'
plt.savefig(name_erro)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:41:49.684109Z","iopub.status.idle":"2021-06-11T14:41:49.684719Z"}}
test_ds_keras = 2.0*test_ds - 1.0
test_masks = model.predict(test_ds_keras)
print(test_masks.shape)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:41:49.686281Z","iopub.status.idle":"2021-06-11T14:41:49.687027Z"}}
len_test_masks = test_masks.shape[0]
# Cria um vetor com 10 valores aleatórios de 0 a 96
randomIndex = np.random.randint(1,len_test_masks,6)

for i in range(0,6,2):
    
    index = randomIndex[i]

    fig, axs = plt.subplots(3,2,figsize = (10, 5))
    plt.subplot(1,2,1)
    plt.imshow(np.squeeze(tests[index]),cmap='gray')
    plt.axis('off')
    plt.xlabel('Base Image')
    
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(test_masks[index]),cmap = 'gray')
    plt.axis('off')
    plt.xlabel("Prediction")
    
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:41:49.688563Z","iopub.status.idle":"2021-06-11T14:41:49.689285Z"}}
pred_train_mask = model.predict(lung_train)
#pred_train_mask = (pred_train_mask > 0.5).astype(np.float32)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.690970Z","iopub.status.idle":"2021-06-11T14:41:49.691871Z"}}
n_images = 3
random_index = np.random.randint(1,pred_train_mask.shape[0],10)

for i in range(n_images):
    
    index = randomIndex[i]
    
    fig, axs = plt.subplots(1,3,figsize=(20,20))
    
    axs[0].imshow(np.squeeze(lung_train[index]),cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Imagem Base')
    
    axs[1].imshow(np.squeeze(mask_train[index]),cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('Máscara')
    
    axs[2].imshow(np.squeeze(pred_train_mask[index]),cmap='gray')
    axs[2].axis('off')
    axs[2].set_title("Predição")

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.693276Z","iopub.status.idle":"2021-06-11T14:41:49.694007Z"}}
def dice_coef_max(y_true, y_pred):
    ''' Dice Coefficient
    Project: BraTs   Author: cv-lee   File: unet.py    License: MIT License
    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    Returns:
        (np.array): Calcula a porcentagem de acerto da rede neural
    '''

    class_num = 1

    for class_now in range(class_num):
    
    # Converte y_pred e y_true em vetores
        y_true_f = K.flatten(y_true[:,:,:])
        y_pred_f = K.flatten(y_pred[:,:,:])

        # Calcula o numero de vezes que
        # y_true(positve) é igual y_pred(positive) (tp)
        intersection = K.sum(y_true_f * y_pred_f)
        # Soma o número de vezes que ambos foram positivos
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        # Smooth - Evita que o denominador fique muito pequeno
        smooth = K.constant(1e-6);
        # Calculo o erro entre eles
        num = (K.constant(2)*intersection + 1)
        den = (union + smooth)
        loss = num / den
        
        if class_now == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    
    total_loss = total_loss / class_num

    return 1 - total_loss

from sklearn.metrics import jaccard_score

def jaccard(x, y):
    x = (x > 0.5).astype(np.bool)
    y = (y > 0.5).astype(np.bool)
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

jac = np.array([])
error = np.array([])

for mask, pred in zip(mask_train,pred_train_mask):
    jac = np.append(jac,jaccard(mask, pred))
    
for mask, pred in zip(mask_train,pred_train_mask):
    error = np.append(error, jaccard(mask, pred))

print('Jaccard desvio padrao {:0.2f}'.format(100*np.std(jac)))
print('Jaccard média {:0.2f}'.format(np.mean(jac)*100))
    
import seaborn as sns
from matplotlib.pyplot import figure

print(f'Total jac:{len(jac)}')
print(f'Menores que 0.943: {(jac < 0.9).sum()}')

figure(figsize=(8, 6), dpi=80)
plt.hist(jac,color='blue',edgecolor='red', bins = int(180/5))
plt.grid()
plt.ylabel('Número de imagens',fontsize=16)
plt.xlabel('Erro da imagem em porcentagem',fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.show()

print('desvio padrao {:0.2f}'.format(100*np.std(error)))
print('Média {:0.2f}'.format(100 - np.mean(error)*100))

plt.plot(range(len(error)),np.sort(error))
    
indexes, erros, acertos = [], [], []
    
for i in range(0,5):
    indexes.append(error.argmin())
    erros.append(error[indexes[i]])
    error[indexes[i]] = 100
print(f'Erros: {erros}')

i = 0
for e in erros:
    print('error {}: {:0.2f}'.format(i, (1 - e)*100))
    i += 1
    
error = np.array([])
for mask, pred in zip(mask_train,pred_train_mask):
    error = np.append(error, jaccard(mask, pred))

max_indexes = []
for i in range(0,5):
    max_indexes.append(error.argmax())
    acertos.append(error[max_indexes[i]])
    error[error.argmax()] = 0
print(acertos)

i = 0
for a in acertos:
    print('acerto {}: {:0.2f}'.format(i,a))
    i += 1

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.695476Z","iopub.status.idle":"2021-06-11T14:41:49.696358Z"}}
def segmentation_lung(lung, mask):
    mask = (mask > 0.8).astype(np.float32)
    seg = lung * mask
    return seg

# %% [code] {"execution":{"iopub.status.busy":"2021-06-11T14:41:49.697748Z","iopub.status.idle":"2021-06-11T14:41:49.698489Z"}}
for index in max_indexes:
        
    plt.imshow(np.squeeze(lung_train[index]),cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.imshow(np.squeeze(mask_train[index]),cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.imshow(np.squeeze(pred_train_mask[index]),cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.imshow(segmentation_lung(lung_train[index],pred_train_mask[index]).reshape((256,256)),cmap='gray')
    plt.axis('off')
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-06-11T14:41:49.700055Z","iopub.status.idle":"2021-06-11T14:41:49.700893Z"}}
n_images = 3
random_index = np.random.randint(1,pred_train_mask.shape[0],10)

from src.output_result.folders import create_folders

create_folders('fig')

for i in range(n_images):
    
    index = randomIndex[i]
    
    plt.imshow(np.squeeze(lung_train[index]),cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.imshow(np.squeeze(mask_train[index]),cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.imshow(np.squeeze(pred_train_mask[index]),cmap='gray')
    plt.axis('off')
    plt.show()