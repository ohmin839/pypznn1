import numpy as np
import matplotlib.pyplot as plt
import pypznn1.deeplearning
from pypznn1.deeplearning import Model
from pypznn1.deeplearning import optimizers
import pypznn1.deeplearning.functions as F
import pypznn1.deeplearning.layers as L

max_epoch = 100
hidden_size = 10
bptt_length = 30

train_set = pypznn1.deeplearning.datasets.SinCurve(train=True)
seqlen = len(train_set)

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def __call__(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

model = SimpleRNN(hidden_size, 1)
optimizer = pypznn1.deeplearning.optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print(f"| epoch {epoch + 1} | loss {avg_loss}")


xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with pypznn1.deeplearning.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.xlabel('y')
plt.legend()
plt.show()
