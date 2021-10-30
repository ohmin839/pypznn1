import numpy as np
from pypznn1.deeplearning import Variable
import pypznn1.deeplearning.functions as F

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.transpose(x)
y.backward(retain_grad=True)
print(x.grad)

