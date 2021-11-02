import numpy as np
import matplotlib.pyplot as plt
from pypznn1.deeplearning import Variable, Model, no_grad
import pypznn1.deeplearning.functions as F
import pypznn1.deeplearning.layers as L

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
l1 = L.Linear(10)
l2 = L.Linear(1)

lr = 0.2
iters = 10000
hidden_size = 10

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = self.l1(x)
        y = F.sigmoid(y)
        y = self.l2(y)
        return y
model = TwoLayerNet(hidden_size, 1)
model.plot(x)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in model.params():
            p.data -= lr * p.grad.data
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
