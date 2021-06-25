from pathlib import Path

from src.dataset.generator_seg import SegmentationDataGenerator as SegDataGen
from src.dataset.dataset_seg import SegmentationDataset
from src.models.segmentacao.segmentacao_model import Unet

DIM = 128

model = Unet(dim=DIM,rate=0.25,activation='relu', final_activation='sigmoid')
model.compile(loss='log_cosh_dice', rf=1)
model.build()
data_path = Path('D:\\Mestrado\\new_data')
dataset = SegmentationDataset(
    path_lung=data_path / 'lungs',
    path_mask=data_path / 'masks'
)
train, val = dataset.partition(val_size=0.2, tamanho=40)

train_generator = SegDataGen(train[0], train[1], batch_size=1, dim=DIM)
val_generator = SegDataGen(val[0], val[1], batch_size=1, dim=DIM)

model.summary()

model.fit(
    x=train_generator,
    validation_data=val_generator,
    epochs=2
)

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

