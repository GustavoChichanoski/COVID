from tensorflow.python.keras.callbacks import Callback
import gc

class ClearGarbage(Callback):

    def __init__(self):
        gc.enable()

    def on_epoch_end(self, epoch, logs):
        gc.collect()