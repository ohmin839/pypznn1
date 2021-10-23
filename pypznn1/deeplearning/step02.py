import numpy as np
from .core import Variable, Function, Square

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
