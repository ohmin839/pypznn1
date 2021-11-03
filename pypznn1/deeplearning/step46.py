import numpy as np
import matplotlib.pyplot as plt
from pypznn1.deeplearning import Variable
from pypznn1.deeplearning import optimizers
from pypznn1.deeplearning import no_grad
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning.models import MLP

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
#optimizer = optimizers.SGD(lr).setup(model)
optimizer = optimizers.MomentumSGD(lr).setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)

# plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
with no_grad():
    y_pred = model(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
