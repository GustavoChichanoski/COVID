from pathlib import Path
import pandas as pd
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow_datasets as tfds

class SegmentatationDatasetGenerator(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """ Dataset metadata (homepage)

    Returns:
        tfds.core.DatasetInfo: retorna o dataset
    """
    return tfds.core.DatasetInfo(
      builder=self,
      features=tfds.features.FeaturesDict({
        'image_description': tfds.features.Text(),
        'image': tfds.features.Image(shape=(256,256,3)),
        'mask': tfds.features.Image(shape=(256,256,1))
      }),
      supervised_keys=('image','image')
    )

  def _split_generators(
    self,
    dl_manager: tfds.download.DownloadManager
  ):
    extract_path = dl_mangager.extract('')
