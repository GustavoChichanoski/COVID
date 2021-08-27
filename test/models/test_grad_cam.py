from functools import reduce
import numpy as np
from src.models.grad_cam_split import div_cuts_per_pixel, sum_grads_cam
import unittest


class TestGradCam(unittest.TestCase):
    def test_div_cuts_per_pixel(self) -> None:

        heatmap = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        div = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        heatmap_divido = div_cuts_per_pixel(div, heatmap)

        esperado = 9.0
        heatmap_divido = reduce(
            lambda a, b: a + b, reduce(lambda a, b: a + b, heatmap_divido)
        )
        self.assertTrue(esperado == heatmap_divido)

    def test_sum_grads_cam(self) -> None:

        heatmap = np.zeros((5, 5))
        split = np.ones((2, 2))
        used_pixels = np.zeros((5, 5))

        grad, pixels = sum_grads_cam(
            grad_cam_full=heatmap,
            grad_cam_cut=split,
            used_pixels=used_pixels,
            start=(2, 2),
        )

        heatmap_test = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        test = grad == heatmap_test
        test_pixel = pixels == heatmap_test

        test_final = True
        for y in test:
            for x in y:
                test_final = test_final and x
        for y in test_pixel:
            for x in y:
                test_final = test_final and x

        self.assertTrue(test_final)