import unittest
import numpy as np
from pypznn1.deeplearning import Variable
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning.utils import gradient_check


class TestBroadcast(unittest.TestCase):

    def test_shape_check(self):
        x = Variable(np.random.randn(1, 10))
        b = Variable(np.random.randn(10))
        y = x + b
        loss = F.sum(y)
        loss.backward()
        self.assertEqual(b.grad.shape, b.shape)