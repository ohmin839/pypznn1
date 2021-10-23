import numpy as np
from .core import Variable

x = Variable(np.array([[1,2,3],[4,5,6]]), 'x')

print(x.name)
print(x.shape)
print(x)

