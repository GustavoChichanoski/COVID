from src.images.data_augmentation import augmentation
from typing import Any, Optional, Tuple
from src.images.read_image import read_images, read_step
from src.images.process_images import augmentation_image
from src.dataset.generator import KerasGenerator
from src.dataset.dataset import Dataset
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

class SegmentationDatasetGenerator(KerasGenerator):

    def __init__(
        self,
        x_set: tfa.types.TensorLike,
        y_set: Optional[tfa.types.TensorLike] == None,
        augmentation: bool = False,
        flip_vertical: bool = False,
        flip_horizontal: bool = False,
        mean_filter: bool = False,
        angle: float = 5.0,
        top: bool = True,
        bot: bool = True,
        right: bool = True,
        left: bool = True,
        load_image_in_ram: bool = False,
        max: float = 1.0,
        min: float = 0.0,
        **params
    ) -> None:
        """Generator to keras fit, it will change the input and output based in `x_set` and `y_set` for each step in train method, it can augmentation input based in atguments `flip_vertical`, to flip batch in vertical axis, `flip_horizontal`, to flip batch in horizontal axis, and `angle`, max angle rotate.

        Args:
            x_set (tfa.types.TensorLike): numpy array contains that path or images of inputs.
            y_set (tfa.types.TensorLike, optional): numpy array contains that path or images of output. Defaults to 0.1.
            augmentation (bool, optional): if augmentation will be applied. Defaults to False.
            flip_vertical (bool, optional): if flip image in vertical will be applied. Defaults to False.
            flip_horizontal (bool, optional): if flip image in horizontal will be applied. Defaults to False.
            mean_filter (bool, optional): if mean filter will be applied. Defaults to False.
            angle (float, optional): max angle to rotate image. Defaults to 5.0.
        """
        super().__init__(
            x_set=x_set,
            y_set=y_set,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            angle=angle,
            **params
        )
        self.mean_filter = mean_filter
        self.augmentation = augmentation
        self.load_image_in_ram = load_image_in_ram
        if self.load_image_in_ram:
            shape = (len(self.x),self.dim,self.dim,self.channels)
            self.x = read_step(self.x,shape)
            self.y = read_step(self.y,shape)
        self.max = max
        self.min = min
        self.top = top
        self.bot = bot
        self.right = right
        self.left = left

    def step_x(
        self,
        batch: tfa.types.TensorLike,
        angle: float = 0
    ) -> Tuple[Any,Any]:
        """
            Get th data of dataset with position initial in idx to idx plus batch_size.

            Args:
                idx (int): initial position

            Returns:
                (Any,Any): the first term is x values of dataset
                           the second term is y vlaues of dataset
        """
        shape = (len(batch), self.dim,self.dim,self.channels)
        if not self.load_image_in_ram:
            batch = read_step(batch, shape)
        batch = (batch / 255.0).astype(np.float32)
        batch = (self.max - self.min) * batch + self.min
        if self.augmentation:
            batch = augmentation(
                batch,
                angle=angle,
                horizontal=self.flip_horizontal,
                vertical=self.flip_vertical,
                top_off=self.top,
                bot_off=self.bot,
                right_off=self.right,
                left_off=self.left
            )

        shape = (int(batch.size / (self.dim * self.dim)),self.dim,self.dim,self.channels)
        batch = batch.reshape(shape)
        return batch

    def step_y(
        self,
        batch: tfa.types.TensorLike,
        angle: float = 0
    ) -> Tuple[Any,Any]:
        """
            Get th data of dataset with position initial in idx to idx plus batch_size.

            Args:
                idx (int): initial position

            Returns:
                (Any,Any): the first term is x values of dataset
                           the second term is y vlaues of dataset
        """
        shape = (len(batch), self.dim,self.dim,self.channels)
        if not self.load_image_in_ram:
            batch = read_step(batch, shape)
        batch = (batch / 255.0).astype(np.float32)
        batch = (self.max - self.min) * batch + self.min
        if self.augmentation:
            batch_out = augmentation(
                batch,
                angle=angle,
                horizontal=self.flip_horizontal,
                vertical=self.flip_vertical
            )
        if self.top:
            batch_out = np.append(batch_out,batch)
        if self.bot:
            batch_out = np.append(batch_out,batch)
        if self.right:
            batch_out = np.append(batch_out,batch)
        if self.left:
            batch_out = np.append(batch_out,batch)
        shape = (int(batch_out.size / (self.dim * self.dim)),self.dim,self.dim,self.channels)
        batch_out = batch_out.reshape(shape)
        return batch_out