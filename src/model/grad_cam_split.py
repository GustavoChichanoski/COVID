"""
    $Module GradCam
"""
import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import Input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.dataset.dataset import zeros
from src.images.process_images import resize_image as resize
from src.images.process_images import normalize_image as norma

def grad_cam_overlay(cuts,
                     positions,
                     model,
                     dim_orig: int = 1024,
                     dim_split: int = 224):
    pb_grad = np.zeros((1024,1024))
    for cut, pos in zip(cuts,positions):
        start = pos[0]
        end = pos[1]
        heatmap  = grad_cam(cut,
                            model)
        pb_grad[start:start + 224,
                end:end + 224] += heatmap
    return pb_grad


def grad_cam(image, model: Model):

    classifier_layer_names = ['max_pool', 'classifier']

    image = image.reshape((1,
                           model.input_shape[1],
                           model.input_shape[1],
                           model.input_shape[-1]))
    predict = model.predict(image)
    target_class = np.argmax(predict[0])

    resnet_model = model.layers[0]

    last_conv_layer = resnet_model.get_layer('post_relu')
    last_conv_layer_model = Model(resnet_model.input, last_conv_layer.output)

    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    layer = classifier_input

    for layer_name in classifier_layer_names:
        try:
            layer = resnet_model.get_layer(layer_name)(layer)
        except:
            layer = model.get_layer(layer_name)(layer)
    classifier_model = Model(classifier_input, layer)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(image)
        tape.watch(last_conv_layer_output)

        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel,
                          last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    heat_max = np.max(heatmap)
    heat_maximum = np.maximum(heatmap,0)

    heatmap = heat_maximum / heat_max
    heatmap = np.array(255*heatmap)
    heatmap = np.uint8(heatmap)
    heatmap = resize(heatmap,224)
    return heatmap
