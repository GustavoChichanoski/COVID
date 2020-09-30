import os
import cv2
import random as rd
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import sklearn.model_selection
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from time import sleep
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import sklearn.model_selection as ms
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from PIL import Image
from keras import backend

class model_segmentation():

    def __init__(
        self,
        img_dim=256,
        path_top="data",
        path_lung="lung",
        path_mask="mask",
        path_test="test"
    ):
        """[Class of CNN to segmentation a chest rays x]

        Args:
            img_dim (int): [Dimension to load the images]. Defaults to 256.
            path_top (str): [top path where find mask lung and test dataset]. Defaults to "data".
            path_lung (str): [path where the images of lung was found]. Defaults to "lung".
            path_mask (str): [path where the images of mask was found]. Defaults to "mask".
            path_test (str): [path where the images of test was found]. Defaults to "test".
        """        
        self.img_size = (img_dim,img_dim)
        self.img_dim = img_dim

        self.path_top = path_top
        self.path_lung = path_lung
        self.path_mask = path_mask
        self.path_test = path_test

        self.model = None


    def model_unet(self):
        """[Return a model of segmentation]

        Returns:
            [model_segmentation]: [class contain model of segmentation]
        """        
        inputs = Input(self.img_dim,self.img_dim,1)
        n      = 32
        neuron = [n,n*2,n*4,n*8,n*16]
        
        conv1 = Conv2D(neuron[0],(3,3),activation='relu',padding='same')(inputs)
        conv1 = Conv2D(neuron[0],(3,3),activation='relu',padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(neuron[1],(3,3),activation='relu',padding='same')(pool1)
        conv2 = Conv2D(neuron[1],(3,3),activation='relu',padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(neuron[2],(3,3),activation='relu',padding='same')(pool2)
        conv3 = Conv2D(neuron[2],(3,3),activation='relu',padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

        conv4 = Conv2D(neuron[3],(3,3),activation='relu',padding='same')(pool3)
        conv4 = Conv2D(neuron[3],(3,3),activation='relu',padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

        conv5 = Conv2D(neuron[4],(3,3),activation='relu',padding='same')(pool4)
        conv5 = Conv2D(neuron[4],(3,3),activation='relu',padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256,(2,2), strides=(2,2),padding='same')(conv5),conv4],axis=3)
        conv6 = Conv2D(neuron[3],(3,3),activation='relu',padding='same')(up6)
        conv6 = Conv2D(neuron[3],(3,3),activation='relu',padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(512,(2,2),strides=(2, 2),padding='same')(conv6),conv3],axis=3)
        conv7 = Conv2D(neuron[2],(3,3),activation='relu',padding='same')(up7)
        conv7 = Conv2D(neuron[2],(3,3),activation='relu',padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64,(2,2),strides=(2, 2),padding='same')(conv7),conv2],axis=3)
        conv8 = Conv2D(neuron[1],(3,3),activation='relu',padding='same')(up8)
        conv8 = Conv2D(neuron[1],(3,3),activation='relu',padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32,(2,2),strides=(2, 2),padding='same')(conv8),conv1],axis=3)
        conv9 = Conv2D(neuron[0],(3,3),activation='relu',padding='same')(up9)
        conv9 = Conv2D(neuron[0],(3,3),activation='relu',padding='same')(conv9)

        conv10 = Conv2D(1,(1,1),activation='sigmoid',padding='same')(conv9)

        return Model(inputs=[inputs], outputs=[conv10])

    def load_weight(self,path='./model/model.h5'):
        """[Load weight of model]

        Args:
            path (str): [Path to weights of model]. Defaults to './model/model.h5'.
        """        
        self.model.load_weight(path)

    def predict(self,image):

        image = (image - 127) / 127
        pred = self.model.predict(image)
        return pred

    def fit(self,training_lung_path,training_mask_path):

        history = []
        weight_path = "./model/{}_weights.best.hdf5".format('cxr_reg')
        checkpoint = ModelCheckpoint(
            weight_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=True
        )

        reduceLROnPlat = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            mode='min',
            epsilon=0.0001,
            cooldown=2,
            min_lr=1e-6
        )

        early = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=15
        )  # probably needs to be more patient, but kaggle time is limited
        callbacks_list = [checkpoint, early, reduceLROnPlat]

        while(training_lung_path != []):
            
            images_mask = []
            images_lung = []
            
            for i in tqdm(range(88)):
                
                if training_lung_path != []:
                    
                    filename_lung = rd.choice(training_lung_path)
                    index = training_lung_path.index(filename_lung)
                    filename_mask = training_mask_path[index]

                    image_lung = cv2.imread(filename_lung)
                    image_lung = cv2.resize(image_lung,self.img_size)
                    image_lung = cv2.bitwise_not(image_lung)
                    image_lung = cv2.cvtColor(image_lung,cv2.COLOR_BGR2GRAY)
                    images_lung.append(image_lung)

                    image_mask = cv2.imread(filename_mask)
                    image_mask = cv2.resize(image_mask,self.img_size)
                    image_mask = cv2.cvtColor(image_mask,cv2.COLOR_BGR2GRAY)
                    images_mask.append(image_mask)

                    rows,cols = image_lung.shape
                    M = np.float32([[1,0,100],[0,1,50]])
                    trans_lung = cv2.warpAffine(image_lung,M,(cols,rows))
                    trans_mask = cv2.warpAffine(image_mask,M,(cols,rows))

                    images_lung.append(trans_lung)
                    images_mask.append(trans_mask)

                    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),10,1)
                    rotate_lung = cv2.warpAffine(image_lung,M,(cols,rows))
                    rotate_mask = cv2.warpAffine(image_mask,M,(cols,rows))

                    images_lung.append(rotate_lung)
                    images_mask.append(rotate_mask)

                    training_lung_path.remove(filename_lung)
                    training_mask_path.remove(filename_mask)
            
                    lung_dataset = np.array(images_lung).reshape(len(images_lung),self.img_dim,self.img_dim,1)
                    mask_dataset = np.array(images_mask).reshape(len(images_mask),self.img_dim,self.img_dim,1)
                    
                    lung_train, lung_validation, mask_train, mask_validation = ms.train_test_split(
                        (lung_dataset - 127.0) / 127.0,
                        (mask_dataset > 127).astype(np.float32),
                        test_size=0.1,
                        random_state=42
                    )
                    
                    history.append(
                        self.model.fit(
                            x=lung_train,
                            y=mask_train,
                            batch_size=1,
                            epochs=50,
                            validation_data=(lung_validation,mask_validation),
                            callbacks=callbacks_list
                        )
                    )