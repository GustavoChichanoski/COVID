import albumentations as A

import cv2

image = cv2.imread('dataset\\dataset\\lungs\\0001.png')
original = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
mask = cv2.imread('dataset\\dataset\\masks\\0001.png')
mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

aug = A.RandomContrast(p=1)
augmented = aug(image=original, mask=mask)

lung = augmented['image']
lung_mask = augmented['mask']

transform = A.Compose([
  A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
  A.HorizontalFlip(p=0.5),
  A.RandomGamma(always_apply=False, p=1.0, gamma_limit=(23, 81), eps=1e-07),
  A.Blur(always_apply=False, p=1.0, blur_limit=(15, 22)),
  A.Equalize(p=0.5),
  A.GaussNoise(var_limit=(200,300), p=0.5),
  A.InvertImg(p=0.5)
])

for i in range(10):
  lung_mask2 = transform(image=original, mask=mask)


  image = lung_mask2['image']
  image2 = lung_mask2['mask']

  print('A')
