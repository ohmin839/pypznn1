import numpy as np
from .core import Variable, add, square

x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(x.grad)

x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)
