import numpy as np
from .core import Variable

x = Variable(np.array(2.0))
y = -x
print(y)

y1 = 2.0 - x
y2 = x - 1.0
print(y1)
print(y2)

y = 3.0 / x
print(y)

y = x ** 3
print(y)

