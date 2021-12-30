import unittest
import cupy as np  # !! CUPY !!
import chainer.functions as CF
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning.utils import gradient_check, array_allclose


class TestPooling(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad, cover_all=False)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 1000
        f = lambda x: F.pooling(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))


class TestAveragePooling(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.average_pooling(x, ksize, stride, pad)
        expected = CF.average_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.average_pooling(x, ksize, stride, pad)
        expected = CF.average_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 1000
        f = lambda x: F.average_pooling(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))