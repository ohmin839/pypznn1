import numpy as np
np.random.seed(0)
from pypznn1.deeplearning import Variable
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning.models import MLP

model = MLP((10, 3))

x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = F.softmax(y)
print(y)
print(p)

x = Variable(np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]]))
t = np.array([2, 0, 1, 0])

y = model(x)
p = F.softmax(y)
print(y)
print(p)

loss = F.softmax_cross_entropy(y, t)
loss.backward()
print(loss)
