import numpy as np
import pypznn1.deeplearning
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning import optimizers
from pypznn1.deeplearning.models import MLP

# hyper parameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = pypznn1.deeplearning.datasets.Spiral(train=True)
test_set = pypznn1.deeplearning.datasets.Spiral(train=False)
train_loader = pypznn1.deeplearning.dataloaders.DataLoader(train_set, batch_size)
test_loader = pypznn1.deeplearning.dataloaders.DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print(f"epoch {epoch+1}")
    print(f"train loss: {sum_loss / len(train_set):.4f}, accuracy: {sum_acc / len(train_set):.4f}")

    sum_loss, sum_acc = 0, 0
    with pypznn1.deeplearning.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(f"test loss: {sum_loss / len(test_set):.4f}, accuracy: {sum_acc / len(test_set):.4f}")
