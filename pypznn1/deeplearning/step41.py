import numpy as np
from pypznn1.deeplearning import Variable
import pypznn1.deeplearning.functions as F

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)
