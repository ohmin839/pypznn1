import unittest
import numpy as np
import chainer.functions as CF
from pypznn1.deeplearning import Variable
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning.utils import gradient_check, array_allclose, array_equal

class TestRelu(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        res = F.relu(x)
        ans = np.array([[0, 0], [2, 0], [0, 1]], np.float32)
        self.assertTrue(array_allclose(res, ans))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])
        self.assertTrue(gradient_check(F.relu, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10) * 100
        self.assertTrue(gradient_check(F.relu, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10) * 100
        self.assertTrue(gradient_check(F.relu, x_data))


class TestLeakyRelu(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        res = F.leaky_relu(x)
        ans = np.array([[-0.2, 0.], [2., -0.6], [-0.4, 1.]], np.float32)
        self.assertTrue(array_allclose(res, ans))

    def test_forward2(self):
        slope = 0.002
        x = np.random.randn(100)
        y2 = CF.leaky_relu(x, slope)
        y = F.leaky_relu(x, slope)
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])
        self.assertTrue(gradient_check(F.leaky_relu, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10) * 100
        self.assertTrue(gradient_check(F.leaky_relu, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10) * 100
        self.assertTrue(gradient_check(F.leaky_relu, x_data))