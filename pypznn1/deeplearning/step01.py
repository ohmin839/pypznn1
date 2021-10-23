import numpy as np
from .variable import Variable

data = np.array(1.0)
x = Variable(data)
print(x.data)
