import unittest
from src.models.metrics.f1_score import F1score
import numpy as np

class TestF1Score(unittest.TestCase):

    def test_f1Score(self):
        m = F1score()
        m.update_state([0,1,1,1],[1,0,1,1])
        resultado = float(m.result().numpy())
        precisao = 2 / 3
        recall = 2 / 3
        f1 = 2 * precisao * recall / (precisao + recall)
        self.assertTrue(abs(resultado - f1) < 1e-5)