import unittest
import numpy as np
import chainer.functions as CF
from pypznn1.deeplearning import Variable
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning.utils import gradient_check, array_allclose


class TestSoftmaxCrossEntropy(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        y = F.softmax_cross_entropy(x, t)
        y2 = CF.softmax_cross_entropy(x, t)
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        f = lambda x: F.softmax_cross_entropy(x, Variable(t))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, CLS_NUM = 10, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: F.softmax_cross_entropy(x, t)
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        N, CLS_NUM = 100, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: F.softmax_cross_entropy(x, t)
        self.assertTrue(gradient_check(f, x))