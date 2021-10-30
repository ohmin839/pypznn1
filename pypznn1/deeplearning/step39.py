import numpy as np
from pypznn1.deeplearning import Variable
import pypznn1.deeplearning.functions as F

x = Variable(np.array([1,2,3,4,5,6]))
y = F.sum()
print(y)
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum()
print(y)
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(axis=0)
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
y = x.sum(keepdims=True)
print(y.shape)
