from functools import reduce
import numpy as np
import unittest

class TestGraph(unittest.TestCase):

  def test_plot_predict_values(self):
    labels = ['Covid', 'Normal', 'Pneumonia']
    predicts = np.random.rand(100,3)
    print(predicts)
