import time
import pypznn1.deeplearning
import pypznn1.deeplearning.functions as F
from pypznn1.deeplearning import optimizers
from pypznn1.deeplearning import DataLoader
from pypznn1.deeplearning.models import MLP

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = pypznn1.deeplearning.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)

#model = MLP((hidden_size, 10))
#optimizer = optimizers.SGD().setup(model)
model = MLP((hidden_size, hidden_size, 10))
optimizer = optimizers.Adam().setup(model)

if pypznn1.deeplearning.cuda.gpu_enable:
    #train_loader.to_cpu()
    #model.to_cpu()
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start
    print(f"epoch {epoch + 1}, loss: {sum_loss / len(train_set)}, time: {elapsed_time}[sec]")
