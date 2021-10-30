import numpy as np
from pypznn1.deeplearning import Variable
import pypznn1.deeplearning.functions as F

x = Variable(np.array([1,2,3,4,5,6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
y = x.sum(keepdims=True)
y.backward()
print(y.shape)
