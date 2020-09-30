# %% [code]
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
from keras import backend as keras

