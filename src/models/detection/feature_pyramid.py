import tensorflow as tf

from keras import Model
from keras.applications.resnet_v2 import ResNet50V2
from keras.layers import Conv2D, UpSampling2D, Layer

def get_backbone() -> Model:
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = ResNet50V2(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

class FeaturePyramid(Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone: Model = None, **kwargs) -> None:
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = Conv2D(filters=256, kernel_size=1, strides=1, padding="same")
        self.conv_c4_1x1 = Conv2D(filters=256, kernel_size=1, strides=1, padding="same")
        self.conv_c5_1x1 = Conv2D(filters=256, kernel_size=1, strides=1, padding="same")
        self.conv_c3_3x3 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.conv_c4_3x3 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.conv_c5_3x3 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.conv_c6_3x3 = Conv2D(filters=256, kernel_size=3, strides=2, padding="same")
        self.conv_c7_3x3 = Conv2D(filters=256, kernel_size=3, strides=2, padding="same")
        self.upsample_2x = UpSampling2D(size=2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output