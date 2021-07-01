# %%
from pathlib import Path

import numpy as np

from src.dataset.generator_seg import SegmentationDatasetGenerator as SegDataGen
from src.dataset.dataset_seg import SegmentationDataset
from src.models.segmentacao.segmentacao_model import Unet

DIM = 512
BATCH_SIZE = 1
TAMANHO_DATASET = 4

model = Unet(dim=DIM,rate=0.25,activation='relu', final_activation='sigmoid')
model.compile(loss='log_cosh_dice', rf=1)
model.build()
model.summary()

data_path = Path('D:\\Mestrado\\new_data')
dataset = SegmentationDataset(
    path_lung=data_path / 'lungs',
    path_mask=data_path / 'masks'
)
train, val = dataset.partition(val_size=0.2, tamanho=TAMANHO_DATASET)

params = {
    'batch_size': BATCH_SIZE,
    'dim': DIM, 'flip_vertical': False,
    'flip_horizontal': False, 'sharpness': True
}
train_generator = SegDataGen(train[0],train[1],augmentation=True,**params)
val_generator = SegDataGen(val[0],val[1],augmentation=True,**params)

model.fit(
    x=train_generator,
    validation_data=val_generator,
    epochs=2
)

# model.save_weights('D:\\Mestrado\\pesos\')
model.load_weights('D:\\Mestrado\\pesos\\pesos.hdf5')

import matplotlib.pyplot as plt

random_index = np.random.randint(10)
predicts = model.predict(train_generator)

# %%
for i in range(len(train_generator)):
    plt.imshow(train_generator[i][0][:][:][0].reshape(DIM,DIM),cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(train_generator[i][1][:][:][0].reshape(DIM,DIM),cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(predicts[i][:][:].reshape(DIM,DIM),cmap='gray')
    plt.axis('off')
    plt.show()

# optimizer = Adam(learning_rate=1e-3)
# loss = DiceError()
# loss_metric = Mean()

# epochs = 100

# for epoch in range(epochs):

#     print(f'Start of epoch {epoch}')

#     # Iterate over the batches of the dataset
#     for step, x_batch_train in enumerate(train_generator):
#         with tf.GradientTape() as tape:
#             batch_x = x_batch_train[0]
#             batch_y = x_batch_train[1]
#             reconstruceted = model(batch_x)
#             # Compute reconstruction loss
#             err = loss(batch_y, reconstruceted)
#             err += sum(model.losses)
        
#         grads = tape.gradient(err, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads,model.trainable_weights))

#         loss_metric(err)

#         if step % 100 == 0:
#             print(f'step {step}: mean loss = {loss_metric.result():.4f}')


# %%
