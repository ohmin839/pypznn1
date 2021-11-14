import numpy as np
from pypznn1.deeplearning import test_mode
import pypznn1.deeplearning.functions as F

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)
