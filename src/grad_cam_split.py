import cv2 as cv
import numpy as np
import imutils
import argparse
import tensorflow as tf
from keras.models import Model


class GradCam:

    def __init__(self,
                 model,
                 classIDX,
                 layerName=None):
        self.model = model
        self.classIdx = classIDX
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        """ Attempt to find the final convolution layer
            by looping over the layers of the network in
            reverse order
        """
        for layer in reversed(self.model.layers):
            # Check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                print(layer.name)
                return layer.name
        # otherwise, we could not find a 4D layer so the
        # GradCAM algorithm cannot be applied
        raise ValueError("Could not find 4D layer.\
                          Cannot apply GradCAM. \
                          Image shape {}".format(layer.output_shape)
                        )

    def compute_heatmap(self,
                        image,
                        eps=1e-8):
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])
        print("Image Shape: {}".format(image.shape))
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_ouput, predictions) = grad_model(inputs)
            loss = predictions[:, self.classIdx]
        grads = tape.gradient(loss, conv_ouput)
        # compute the guided gradients
        cast_conv_ouputs = tf.cast(conv_ouput > 0,
                                   "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = (cast_conv_ouputs
                       * cast_grads
                       * grads)

        # The convolution and guided gradients have a
        # batch normalization (which we don't need) so
        # let's grab the volume itself and discard the
        # batch
        conv_ouput = conv_ouput[0]
        guided_grads = guided_grads[0]

        # compute the average of the gradient values and
        # using them as weights, compute the ponderation
        # of the fitlers with respect to the weights
        weights = tf.reduce_mean(guided_grads,
                                 axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights,
                                        conv_ouput),
                            axis=-1)
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie
        # in the range [0,1], scale the resulting values
        # to the range [0,255], and then convert to an
        # unsigned 8-bit integer
        num = heatmap - np.min(heatmap)
        den = (heatmap.max() - heatmap.min()) + eps
        heatmap = num / den
        heatmap = (heatmap * 255).astype("unit8")

        # Return the resulting heatmap to the calling
        # function
        return heatmap

    def overlay_heatmap(self,
                        heatmap,
                        image,
                        alpha=0.5,
                        colormap=cv.COLORMAP_PLASMA):
        """ apply the supplied color mpa to the heatmap and
        then overlay the heatmap on the input image """
        heatmap = cv.applyColorMap(heatmap, colormap)
        output = cv.addWeighted(image,
                                alpha,
                                heatmap,
                                1 - alpha,
                                0)
        # Return a 2-tuple of the color mapped heatmap and the output overlaid image
        return (heatmap, output)
