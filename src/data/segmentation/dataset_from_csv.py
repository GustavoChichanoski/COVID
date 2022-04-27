# %% import 
import tensorflow as tf
import os
import tensorflow_datasets as tfds

# %% Codigo de inicialização deve estar no inicio do programam
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
# this is the initialization code that has to be at the beginning
tf.tpu.experimental.initialize_tpu_system(resolver)
print('All devices: ', tf.config.list_logical_devices('TPU'))

# %% Estrategia de distribuição
strategy = tf.distribute.TPUStrategy(resolver)

