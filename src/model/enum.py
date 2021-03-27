from enum import Enum, unique

@unique
class AvailableModels(Enum):
    RESNET = 'Resnet50V2'
    VGG = 'VGG19'
    DENSENET = 'DenseNet201'
    INCEPTION = 'InceptionResNetV2'