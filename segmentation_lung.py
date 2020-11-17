# %% [code]
import os
import cv2 as cv
import h5py
import numpy as np  # linear algebra
import random as rd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
from tqdm import tqdm
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
from keras.regularizers import l1_l2
from keras.preprocessing import image_dataset_from_directory
%matplotlib inline

# %% [code]
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] 
    # to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

# %% [code]
def grayscale(image):
    const_gray = cv.COLOR_BGR2GRAY
    return cv.cvtColor(image,const_gray)

def rescale_image(image,scale=255):
    half_scale = scale/2
    return (image - half_scale)/half_scale
    
def equalize_histogram(image):
    return cv.equalizeHist(image)

def invert_image(image):
    """
        Inverte as cores da imagem
        Ex: (0 vira 255) e (255 vira 0)
        Args:
            imagem - np.array
        return
            imagem - np.array
    """
    return cv.bitwise_not(image)

def read_image(url):
    """
        Lê a imagem passada pela url e retorna a
        imagem em BGR com o tamanho real
        Args:
            url - Tipo string: "./img.png"
        return:
            imagem do tipo np.array
    """
    return cv.imread(url)

def image_resize(image,size=(256,256)):
    """
    Redimensiona a imagem para o tamanho do parametro size.
        Args:
            image - Deve ser do tipo np.array
            size - Deve ser uma tupla (w,h) onde
                   w é a largura e h é altura
        Return
            Retorna a imagem com tamanho (w,h)
    """
    return cv.resize(image,size)

def lungs_images(lungs_path,input_shape):
    lungs = []
    for lung_path in tqdm(lungs_path):
        lung = read_image(lung_path)
        lung = image_resize(lung,input_shape)
        lung = grayscale(lung)
        lung = adjust_gamma(lung,0.5)
        lung = rescale_image(lung)
        lungs.append(lung)
    return lungs

def masks_images(masks_path,input_shape):
    masks = []
    for mask_path in tqdm(masks_path):
        mask = read_image(mask_path)
        mask = image_resize(mask,input_shape)
        mask = grayscale(mask)
        mask = rescale_image(mask)
        masks.append(mask)
    return masks

def get_data(
    input_shape=(256,256),
    path_train_lung ='data/lung', 
    path_train_mask ='data/masks',
    path_test_lung = 'data/test'):
    """
        image_shape: Tamanho da imagem para ser rescalada
        path_train_lung : caminhos completos dos arquivos de 
        treino das imagens dos pulmões 
        path_train_mask : caminhos completos dos arquivos de 
        treino das mascaras dos pulmões
        path_test : caminhos completos dos arquivos de testes 
        das imagens dos pulmões
    """
    lungs = lungs_images(path_train_lung,input_shape)
    tests = lungs_images(path_test_lung,input_shape)
    masks = masks_images(path_train_mask,input_shape)
    return lungs, masks, tests

# %% [code]
def conv_unet(
    layer,
    filters=32,kernel=(3,3),
    act="relu",
    i=1,j=1):
    # Define os nomes das layers
    conv_name = "Conv{}_{}".format(i,j)
    bn_name = "BN{}_{}".format(i,j) 
    act_name = "Act{}_{}".format(i,j)

    layer = Conv2D(
        filters=filters,kernel_size=kernel,
#         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
        padding='same', name=conv_name)(layer)
#     layer = BatchNormalization(name=bn_name)(layer)
    layer = Activation(act,name=act_name)(layer)
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

# %% [code]
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
        
    layer = Dropout(0.33,name='Drop_1')(layer)
    outputs = Conv2D(
        n_class,(1,1),padding='same',
        activation=final_activation, name='output'
    )(layer)
    
    return Model(inputs,outputs,name="UNet")

# %% [code]
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

# %% [code]
def fill_subplot(axs,x,y1,y2,legend,title):
    axs.plot(x,y1,'r--')
    axs.plot(x,y2,'b--')
    axs.legend(legend)
    axs.set_title(title)
    return axs

# %% [code]
def get_path_images(path_lung,path_mask,path_test):
    lungs, masks, tests = [], [], []
    masks_in_path_mask = os.listdir(path_mask)
    for mask_id in masks_in_path_mask:
        if mask_id.find('CHN'):
            full_path = os.path.join(path_lung,mask_id)
            lungs.append(full_path)
        else:
            path = mask_id.split("_mask.png")[0]
            path = '{}.png'.format(path)
            full_path = os.path.join(path_lung,path)
            lungs.append(full_path)
        full_path = os.path.join(path_mask,mask_id)
        masks.append(full_path)
    for test_id in os.listdir(path_test):
        full_path = os.path.join(path_test,test_id)
        tests.append(full_path)
    return lungs, masks, tests

# %% [code]
DIM = 256
EPOCH = 100
IMG_SIZE = (DIM,DIM)
PATH = '../input/chest-xray-masks-and-labels/Lung Segmentation'
PATH_MODEL = 'model.h5'
BATCH_SIZE = 10

# %% [code]
lung_path = os.path.join(PATH,'CXR_png')
mask_path = os.path.join(PATH,'masks')
test_path = os.path.join(PATH,'test')
weight_path = "{}_weights.best.hdf5".format('cxr_reg')

# %% [code]
train_lung_path, train_mask_path, test_lung_path = get_path_images(
    lung_path,
    mask_path,
    test_path)

# %% [code]
metrics = [TruePositives(name='tp'),  # Valores realmente positivos
           TrueNegatives(name='tn'),  # Valores realmente negativos
           FalsePositives(name='fp'), # Valores erroneamente positivos
           FalseNegatives(name='fn'), # Valores erroneamente negativos
           BinaryAccuracy(name='accuracy')]

# %% [code]
filtros = 32
depth = 5
act = 'elu'
# Criação e compilação do modelo 1 proposto
model = model_unet((DIM,DIM,1),filter_root=filtros,depth=depth,activation=act)
model.compile(
    optimizer = Adam(lr=1e-3),
    loss = dice_coef_loss,
    metrics = metrics)
model.summary()

# %% [code]
from keras.utils import plot_model
plot_model(model, "my_first_model.png",show_shapes=True)

# %% [code]
# Metrica de salvamento
checkpoint = ModelCheckpoint(
    weight_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=True)
# Metrica para a redução do valor de LR
reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    mode='min',
    epsilon=1e-2,
    cooldown=2,
    min_lr=1e-8)
# Metrica para a parada do treino
early = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=40)
# Vetor a ser passado na função fit
callbacks_list = [checkpoint, early, reduceLROnPlat]

# %% [code]
# Fixa a aleatoridade do numpy
np.random.seed(42)
lungs, masks, tests = get_data(
    input_shape=(256,256),
    path_train_lung = train_lung_path, 
    path_train_mask = train_mask_path,
    path_test_lung = test_lung_path)
# %% [code]
def numpy_to_keras(nparray,dim):
    keras = np.array(nparray).reshape(len(nparray),dim,dim,1)
    return keras
lung_ds = numpy_to_keras(lungs,DIM)
mask_ds = numpy_to_keras(masks,DIM)
test_ds = numpy_to_keras(tests,DIM)

# %% [code]
lung_train, lung_val, mask_train, mask_val = ms.train_test_split(
    lung_ds,
    (mask_ds > 0).astype(np.float32),
    test_size=0.2, random_state=42)

# %% [code]
history = []

# %% [code]
# Criação e compilação do modelo 1 proposto
history.append(
    model.fit(
        x=lung_train,
        y=mask_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        validation_data=(lung_val,mask_val),
        callbacks=callbacks_list,
        workers=3))

# %% [code]
training_loss, validation_loss = [], []
training_accuracy, validation_accuracy = [], []

training_tp, validation_tp = [], []
training_tn, validation_tn = [], []
training_fp, validation_fp = [], []
training_fn, validation_fn = [], []

total = (EPOCH*DIM*DIM)

for i in history:

    training_loss += i.history['loss']
    validation_loss += i.history['val_loss']
    
    training_accuracy += i.history['accuracy']
    validation_accuracy += i.history['val_accuracy']
    
    training_tp += i.history['tp']
    validation_tp += i.history['val_tp']
    
    training_fp += i.history['fp']
    validation_fp += i.history['val_fp']
    
    training_tn += i.history['tn']
    validation_tn += i.history['val_tn']
    
    training_fn += i.history['fn']
    validation_fn += i.history['val_fn']

# %% [code]
val_acc = (1 - validation_loss[-1])*100
model.save(
    'model_acc_{:.02f}.h5'.format(val_acc),
    overwrite=True)
file_name_weight = 'weights_N_{}_depth_{}_act_{}_acc_{:.02f}.hdf5'.format(i,j,act,val_acc).format(val_acc)
model.save_weights(
    file_name_weight,
    overwrite=False)

# %% [code]
fig, axs = plt.subplots(3,2,figsize=(10,10))
   
epoch_count = range(1,len(training_loss) + 1)

str_val, str_train = "Validation ", "Training "
str_acc, str_loss = "Accuracy ", "Loss "
str_p, str_n = "Positives ", "Negatives "
str_f, str_t = "False ", "True "

str_train_loss = "{}{}".format(str_train,str_loss)
str_val_loss = "{}{}".format(str_val,str_loss)

str_train_acc = "{}{}".format(str_train,str_acc)
str_val_acc = "{}{}".format(str_val,str_acc)

str_fp = "{}{}".format(str_f,str_p)
str_fn = "{}{}".format(str_f,str_n)

str_tp = "{}{}".format(str_t,str_p)
str_tn = "{}{}".format(str_t,str_n)

str_train_fn = "{}{}".format(str_train,str_fn)
str_train_fp = "{}{}".format(str_train,str_fp)

str_val_fn = "{}{}".format(str_val,str_fn)
str_val_fp = "{}{}".format(str_val,str_fp)

str_train_tn = "{}{}".format(str_train,str_tn)
str_train_tp = "{}{}".format(str_train,str_tp)

str_val_tn = "{}{}".format(str_val,str_tn)
str_val_tp = "{}{}".format(str_val,str_tp)

axs[0,0] = fill_subplot(axs=axs[0,0],
    x=epoch_count, title=str_loss,
    y1=training_loss,y2=validation_loss,
    legend=[str_train_loss,str_val_loss])
axs[0,1] = fill_subplot(axs=axs[0,1],
    x=epoch_count,title=str_acc,
    y1=training_accuracy,y2=validation_accuracy,
    legend=[str_train_acc,str_val_acc])
axs[1,0] = fill_subplot(axs=axs[1,0],
    x=epoch_count,
    y1=training_tp,y2=training_fp,
    legend=[str_train_fp,str_train_tp],
    title='Trainig Positives: True vs False')
axs[1,1] = fill_subplot(axs=axs[1,1],
    x=epoch_count,
    y1=validation_tp,y2=validation_fp,
    legend=[str_val_tp,str_val_fp],
    title='Validation Positives: True vs False')
axs[2,0] = fill_subplot(axs=axs[2,0],
    x=epoch_count,
    y1=training_tn,y2=training_fn,
    legend=[str_train_tn,str_train_fn],
    title='Trainig Negatives: True vs False')
axs[2,1] = fill_subplot(axs=axs[2,1],
    x=epoch_count,
    y1=validation_tn,y2=validation_fn,
    legend=[str_val_tn,str_val_fn],
    title='Validation Negatives: True vs False')

# %% [code]
test_masks = model.predict(test_ds)
print(test_masks.shape)

# %% [code]
len_test_masks = test_masks.shape[0]
# Cria um vetor com 10 valores aleatórios de 0 a 96
randomIndex = np.random.randint(1,len_test_masks,6)
fig, axs = plt.subplots(3,2,figsize = (10, 10))

for i in range(0,6,2):
    
    index = randomIndex[i]
    
    plt.subplot(3,2,i+1)
    plt.imshow(np.squeeze(tests[index]),cmap='gray')
    plt.axis('off')
    plt.xlabel('Base Image')
    
    plt.subplot(3,2,i+2)
    plt.imshow(np.squeeze(test_masks[index]),cmap = 'gray')
    plt.axis('off')
    plt.xlabel("Prediction")

# %% [code]
pred_train_mask = model.predict(lung_train)
#pred_train_mask = (pred_train_mask > 0.5).astype(np.float32)

# %% [code]
n_images = 5
random_index = np.random.randint(1,pred_train_mask.shape[0],10)
fig, axs = plt.subplots(n_images,3,figsize=(10,10))

for i in range(n_images):
    
    index = randomIndex[i]
    
    axs[i,0].imshow(np.squeeze(lung_train[index]),cmap='gray')
    axs[i,0].axis('off')
    axs[i,0].set_title('Base Image')
    
    axs[i,1].imshow(np.squeeze(mask_train[index]),cmap='gray')
    axs[i,1].axis('off')
    axs[i,1].set_title('Mask Image')
    
    axs[i,2].imshow(np.squeeze(pred_train_mask[index]),cmap = 'gray')
    axs[i,2].axis('off')
    axs[i,2].set_title("Prediction")

# %% [code]
acuracia = model.evaluate(x=lung_train,y=mask_train)
print(acuracia)

# %% [code]
# Modelo tipo 2

# %% [code]
def split_image_in_K_numbers(lung,mask,n_split,dim_orig,dim_split):

    splits_lung, splits_mask = [], []

    for i in range(n_split):

        # Sorteia x e y aleatoriamente de 0 a dim_orig - dim_split - 1
        x = np.random.randint(0,dim_orig - dim_split)
        y = np.random.randint(0,dim_orig - dim_split)

        # Define o valor máximo de x e y
        x_end = x+dim_split
        y_end = y+dim_split

        # Corta a image de x a x_end
        split_lung = lung[x:x_end, y:y_end]
        split_mask = mask[x:x_end, y:y_end]

        # Armazena em um vetor
        splits_lung.append(split_lung)
        splits_mask.append(split_mask)
        
    return splits_lung, splits_mask

# %% [code]
lung_url = "../input/chest-xray-masks-and-labels/Lung Segmentation/CXR_png/CHNCXR_0001_0.png"
mask_url = "../input/chest-xray-masks-and-labels/Lung Segmentation/masks/CHNCXR_0001_0_mask.png"
DIM_ORIG = 1024
DIM_SPLIT = 224
K_SPLIT = 111

lung = cv.imread(lung_url)
lung = cv.resize(lung,(DIM_ORIG,DIM_ORIG))
mask = cv.imread(mask_url)
mask = cv.resize(mask,(DIM_ORIG,DIM_ORIG))

splits_lung, splits_mask = split_image_in_K_numbers(
    lung,mask,K_SPLIT,DIM_ORIG,DIM_SPLIT)

# %% [code]
n_images = 5
fig, axs = plt.subplots(n_images,2,figsize=(10,10))
for i in range(n_images):
    
    random_index = np.random.randint(100)
    
    axs[i,0].imshow(splits_lung[random_index],cmap='gray')
    axs[i,0].axis('off')
    axs[i,0].set_title('Split lung image {}'.format(random_index))
    
    axs[i,1].imshow(splits_mask[random_index],cmap='gray')
    axs[i,1].axis('off')
    axs[i,1].set_title('Split mask image {}'.format(random_index))

# %% [code]
model2 = model_unet(
    (DIM_SPLIT,DIM_SPLIT,1),5,'relu',1,'sigmoid',32)
model2 = model_unet((DIM,DIM,1),filter_root=8,depth=5,activation='elu')
model2.compile(
    optimizer = Adam(lr=1e-3),
    loss = dice_coef_loss,
    metrics = metrics)

# %% [code]
# Arrays com os caminhos completos das imagens
# train_lung_path, train_mask_path, test_lung_path
history = []

for index in tqdm(range(0,len(train_lung_path),7)):
    
    lungs, masks = [], []
    
    for i in tqdm(range(10)):
        
        if i < len( train_mask_path):

            mask_path = train_mask_path[index + i]
            lung_path = train_lung_path[index + i]

            mask = read_image(mask_path)
            mask = image_resize(mask,(DIM_ORIG,DIM_ORIG))
            mask = grayscale(mask)
            mask = rescale_image(mask)

            lung = read_image(lung_path)
            lung = image_resize(lung,(DIM_ORIG,DIM_ORIG))
            lung = grayscale(lung)
            lung = adjust_gamma(lung,0.5)
            lung = rescale_image(lung)

            splits_lung, splits_mask = split_image_in_K_numbers(
                lung,mask,K_SPLIT,DIM_ORIG,DIM_SPLIT)

            for lung in splits_lung:
                lungs.append(lung)
            for mask in splits_mask:
                masks.append(mask)
                
        lung_ds = numpy_to_keras(lungs,DIM_SPLIT)
        mask_ds = numpy_to_keras(masks,DIM_SPLIT)

    lung_train, lung_val, mask_train, mask_val = ms.train_test_split(
        (2.0*lung_ds - 1.0),
        (mask_ds > 0.5).astype(np.float32),
        test_size=0.2, random_state=42)

    history.append(
        model2.fit(x=lung_train,y=mask_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        validation_data=(lung_val,mask_val),
        callbacks=callbacks_list))

# %% [code]
lung = read_image(lung_url)
lung = image_resize(lung,(DIM_ORIG,DIM_ORIG))
lung = grayscale(lung)
lung = adjust_gamma(lung,0.5)
lung = rescale_image(lung)

mask = read_image(mask_url)
mask = image_resize(mask,(DIM_ORIG,DIM_ORIG))
mask = grayscale(mask)
mask = rescale_image(mask)

splits_lung, splits_mask = [], []

# Sorteia x e y aleatoriamente de 0 a dim_orig - dim_split - 1
x = np.random.randint(0,DIM_ORIG - DIM_SPLIT,K_SPLIT)
y = np.random.randint(0,DIM_ORIG - DIM_SPLIT,K_SPLIT)

for i in range(K_SPLIT):
    
    # Define o valor máximo de x e y
    x_end = x[i]+DIM_SPLIT
    y_end = y[i]+DIM_SPLIT

    # Corta a image de x a x_end
    split_lung = lung[x[i]:x_end, y[i]:y_end]
    split_mask = mask[x[i]:x_end, y[i]:y_end]

    # Armazena em um vetor
    splits_lung.append(split_lung)
    splits_mask.append(split_mask)

lung_k = numpy_to_keras(splits_lung,DIM_SPLIT)
split_predict = model2.predict(lung_k)
mask_predict = np.zeros((DIM_ORIG,DIM_ORIG))
mask_real = np.zeros((DIM_ORIG,DIM_ORIG))

for index in range(K_SPLIT):
    
    x_orig = x[index]
    y_orig = y[index]
    
    x_end = x_orig + DIM_SPLIT
    y_end = y_orig + DIM_SPLIT
    
    mask_real[x_orig:x_end,y_orig:y_end] += splits_mask[index]
    mask_predict[x_orig:x_end,y_orig:y_end] += lung_k[index].reshape(DIM_SPLIT,DIM_SPLIT)
    
plt.imshow((mask_predict.reshape(DIM_ORIG,DIM_ORIG).astype(np.float32) > 2),cmap='gray')
plt.show()

# %% [code]
plt.imshow((mask_real > 1).astype(np.float32),cmap='gray')
plt.show()