import os
import time
import gzip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

num_epochs = 100
batch_size = 32
learning_rate = 3e-3

pre_train = False
root_dir = os.path.expanduser('~/dataset/fashion_mnist')

class FashionMnistDataset(Dataset):
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
        self._data, self._target = self._read_data(image_file, label_file)

    def __getitem__(self, idx):
        return self._data[idx].div(255).float(), self._target[idx].long()

    def __len__(self):
        return len(self._target)

    def _read_data(self, image_file, label_file):
        with gzip.open(label_file, 'rb') as f:
            labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        with gzip.open(image_file, 'rb') as f:
            images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
        return torch.from_numpy(np.array(images)), torch.from_numpy(np.array(labels))

class CNNModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # (N, 10, 24, 24)
        self.pool1 = nn.MaxPool2d(2)
        # (N, 10, 12, 12)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # (N, 20, 8, 8)
        self.pool2 = nn.MaxPool2d(2)
        # (N, 20, 4, 4)
        self.linear1 = nn.Linear(20*4*4, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = CNNModel(10)

loss_iter, acc_iter = [], []

trainset = FashionMnistDataset(root_dir, True)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=16)

testset = FashionMnistDataset(root_dir, False)
testloader = DataLoader(dataset=testset, batch_size=batch_size, num_workers=16)

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
    test_acc /= len(dataloader.dataset) / 100
    acc_iter.append(test_acc)
    print(f'Test Error: acc={test_acc:>0.3f}, loss={test_loss:>7f}')

def save_checkpoint(filename, model, optimizer):
    print(f'Save model checkpoint to {filename}')
    checkpoint = {
        'state': model.state_dict(),
        'optim': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    print(f'Load model checkpoint from {filename}')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state'])
    optimizer.load_state_dict(checkpoint['optim'])

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

if pre_train:
    load_checkpoint(model, optimizer, 'model.pth')

for epoch in range(num_epochs):
    print(f'Epoch {epoch} {"":->60s}')
    start = time.time()
    train(trainloader, model, criterion, optimizer)
    time1 = time.time()
    test(testloader, model, criterion)
    time2 = time.time()
    print(f'Time Cost: train={time1-start:.1f}s, test={time2-time1:.1f}s')
    if (epoch+1) % 10 == 0:
        save_checkpoint('model.pth', model, optimizer)

plot_loss(loss_iter, acc_iter)
