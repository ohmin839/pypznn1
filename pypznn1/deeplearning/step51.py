import pypznn1.deeplearning
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning import optimizers
from pypznn1.deeplearning import DataLoader
from pypznn1.deeplearning.models import MLP

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = pypznn1.deeplearning.datasets.MNIST(train=True)
test_set = pypznn1.deeplearning.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

#model = MLP((hidden_size, 10))
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
#optimizer = optimizers.SGD().setup(model)
optimizer = optimizers.Adam().setup(model)

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

    print(f"epoch: {epoch + 1}")
    print(f"train loss: {sum_loss / len(train_set):.4f}, accuracy: {sum_acc / len(train_set):.4f}")

    sum_loss, sum_acc = 0, 9
    with pypznn1.deeplearning.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(f"test loss: {sum_loss / len(test_set):.4f}, accuracy: {sum_acc / len(test_set):.4f}")
