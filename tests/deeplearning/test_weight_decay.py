import unittest
import numpy as np
import pypznn1.deeplearning
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning.utils import array_allclose


class TestWeightDecay(unittest.TestCase):

    def test_compare1(self):
        rate = 0.4
        x = np.random.rand(10, 2)
        t = np.zeros((10)).astype(int)
        layer = pypznn1.deeplearning.layers.Linear(in_size=2, out_size=3, nobias=True)
        layer.W.data = np.ones_like(layer.W.data)
        optimizer = pypznn1.deeplearning.optimizers.SGD().setup(layer)
        optimizer.add_hook(pypznn1.deeplearning.optimizers.WeightDecay(rate=rate))

        layer.cleargrads()
        y = layer(x)
        y = F.softmax_cross_entropy(y, t)
        y.backward()
        optimizer.update()
        W0 = layer.W.data.copy()

        layer.W.data = np.ones_like(layer.W.data)
        optimizer.hooks.clear()
        layer.cleargrads()
        y = layer(x)
        y = F.softmax_cross_entropy(y, t) + rate / 2 * (layer.W ** 2).sum()
        y.backward()
        optimizer.update()
        W1 = layer.W.data
        self.assertTrue(array_allclose(W0, W1))
