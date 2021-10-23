import numpy as np
from .core import Variable, Square, Exp
from .utils import numerical_diff

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)

