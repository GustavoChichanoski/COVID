import os
from tqdm import tqdm
import cv2
import numpy as np

def process_image(
    image=None,
    is_lung=True,
    input_shape=(256,256),
    path=None,
    index=0):
    """
    Process a image in numpy array to resize and transform in grayscale
    Example use:
        image = process_image(image)
    """
    image = cv2.resize(image, input_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if is_lung:
        image = cv2.bitwise_not(image)

    save_file = os.path.join(path,'{:04d}.png'.format(index))
    cv2.imwrite(save_file,image)

def process_mask_lung(
    image_lung=None,
    image_mask=None,
    path_lung='old_data/lung',
    path_mask='old_data/mask',
    input_shape=(256,256),
    index=0
):
    process_image(
        image=image_lung,
        input_shape=input_shape,
        path=path_lung,
        index=index
    )
    process_image(
        image=image_mask,
        is_lung=False,
        input_shape=input_shape,
        path=path_mask,
        index=index
    )

def image_path_to_array(
    path_lung='old_data/lung',
    path_mask='old_data/mask',
    path_test='old_data/test',
    dim=256):
    """
    Return a numpy array of images in a new path
    resize and in grayscale
    use example:
    image_array = image_path_to_array('data/masks')
    """
    index = 0
    input_shape=(dim,dim)
    path_mask_images = os.listdir(path=path_mask)

    for path_mask_image in tqdm(path_mask_images):
        
        full_path_mask = os.path.join(path_mask,path_mask_image)
        
        if path_mask_image.find('CHN'):
            full_path_lung = os.path.join(path_lung,path_mask_image)
        else:
            full_path_lung = os.path.join(
                path_lung,
                '{}.png'.format(path_mask_image.split('_mask.png')[0])
            )
        
        image_lung = cv2.imread(full_path_lung)
        image_mask = cv2.imread(full_path_mask)
        
        if image_lung is not None:
            
            process_mask_lung(
                image_lung=image_lung,
                image_mask=image_mask,
                path_lung='data/lung',
                path_mask='data/mask',
                input_shape=input_shape,
                index=index
            )
            index += 1
            # Translation
            rows,cols,_ = image_lung.shape
            M = np.float32([[1,0,100],[0,1,50]])
            translation_lung = cv2.warpAffine(image_lung,M,(cols,rows))
            translation_mask = cv2.warpAffine(image_mask,M,(cols,rows))

            process_mask_lung(
                image_lung=translation_lung,
                image_mask=translation_mask,
                path_lung='data/lung',
                path_mask='data/mask',
                input_shape=input_shape,
                index=index
            )
            index += 1

            # Rotate
            M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),10,1)
            translation_lung = cv2.warpAffine(image_lung,M,(cols,rows))
            translation_mask = cv2.warpAffine(image_mask,M,(cols,rows))

            process_mask_lung(
                image_lung=translation_lung,
                image_mask=translation_mask,
                path_lung='data/lung',
                path_mask='data/mask',
                input_shape=input_shape,
                index=index
            )
            index += 1

    path_test_images = os.listdir(path=path_test)
    new_index = 0

    for path_test_image in tqdm(path_test_images):

        image_test = cv2.imread(os.path.join(path_test,path_test_image))
        process_image(
            image=image_test,
            input_shape=input_shape,
            path='data/test',
            index=new_index
        )
        new_index += 1

PATH_DATA = 'old_data'

old_path_lung = os.path.join(PATH_DATA,'lung')
old_path_mask = os.path.join(PATH_DATA,'mask')
old_path_test = os.path.join(PATH_DATA,'test')

image_path_to_array(
    old_path_lung,
    old_path_mask,
    old_path_test
)