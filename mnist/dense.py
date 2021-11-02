import os
import gzip
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

num_epochs = 10
batch_size = 32
learning_rate = 1e-2

root_dir = os.path.expanduser('~/dataset/mnist')

class MnistDataset(Dataset):
    def __init__(self, root='.', train=True):
        super().__init__()
        self._root = root
        self._train = train
        if self._train:
            image_file = os.path.join(root, 'train-images-idx3-ubyte.gz')
            label_file = os.path.join(root, 'train-labels-idx1-ubyte.gz')
        else:
            image_file = os.path.join(root, 't10k-images-idx3-ubyte.gz')
            label_file = os.path.join(root, 't10k-labels-idx1-ubyte.gz')
        self._x, self._y = self._read_data(image_file, label_file)

    def __getitem__(self, idx):
        return self._x[idx].div(255).float().view(-1), self._y[idx].long()

    def __len__(self):
        return len(self._y)

    def _read_data(self, image_file, label_file):
        with gzip.open(label_file, 'rb') as f:
            labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        with gzip.open(image_file, 'rb') as f:
            images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return torch.from_numpy(images), torch.from_numpy(labels)

class LinearModel(torch.nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, 512)
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return self.linear4(x)

model = LinearModel(784, 10)

loss_iter, acc_iter = [], []

trainset = MnistDataset(root_dir, True)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=8)

testset = MnistDataset(root_dir, False)
testloader = DataLoader(dataset=testset, batch_size=batch_size, num_workers=8)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for idx, (x, y) in enumerate(dataloader):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            loss, current = loss.item(), idx * len(x)
            print(f'loss={loss:>7f} [{current:>5d}/{len(dataloader.dataset):>5d}]')

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for (x, y) in dataloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            acc = (y_pred.argmax(1)==y).float().sum()
            test_acc += acc.item()
    test_loss /= len(dataloader)
    loss_iter.append(test_loss)
    test_acc /= len(dataloader.dataset)
    acc_iter.append(test_acc)
    print(f'Test Error: acc={100*test_acc:>0.3f}, loss={test_loss:>7f}')

def plot_loss(loss, acc):
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(len(loss)), loss, label='Test Loss')
    plt.subplot(212)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(range(len(acc)), acc, label='Test Acc')
    plt.savefig('loss.png')

for epoch in range(num_epochs):
    print(f'Epoch {epoch} {"":->60s}')
    train(trainloader, model, criterion, optimizer)
    test(testloader, model, criterion)

plot_loss(loss_iter, acc_iter)
