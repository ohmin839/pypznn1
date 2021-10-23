import numpy as np
from .core import Variable, square, exp

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

x = Variable(np.array(1.0)) #OK
x = Variable(None) #OK
x = Variable(1.0) #NG
