import numpy as np
from .core import Variable, add, square
from .core import using_config, no_grad

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)

with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)

x = Variable(np.array(2.0))
y = square(x)
y.backward()
print(y.data, x.grad)

