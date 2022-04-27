from functools import reduce
import numpy as np
import unittest
from src.plots.graph import plot_predict_values

class TestGraph(unittest.TestCase):

  def test_plot_predict_values(self):
    labels = ['Covid', 'Normal', 'Pneumonia']
    predicts = np.random.rand(100,3)
    print(predicts)
    plot_predict_values(predicts)