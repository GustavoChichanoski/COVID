import tensorflow_addons as tfa
from tensorflow_addons import TensorLike as tensor

def normalize(image: tensor, mean: int = 128, std: int = 128) -> tensor:
    """normalize the SSD input and return it

    Args:
        image (tensor): [description]
        mean (int, optional): [description]. Defaults to 128.
        std (int, optional): [description]. Defaults to 128.

    Returns:
        tensor: [description]
    """   
    image = (image * 256 - mean) / std
    return image

