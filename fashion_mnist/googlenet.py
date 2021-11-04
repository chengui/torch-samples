import os
import gzip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from torch.utils.data import Dataset, DataLoader

num_epochs = 10
batch_size = 16
learning_rate = 1e-3

pre_train = False
root_dir = os.path.expanduser('~/dataset/fashion_mnist')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        data, target = self._data[idx], self._target[idx]
        data = VF.resize(data, (96, 96))
        return data.div(255).float(), target.long()

    def __len__(self):
        return len(self._target)

    def _read_data(self, image_file, label_file):
        with gzip.open(label_file, 'rb') as f:
            labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        with gzip.open(image_file, 'rb') as f:
            images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 1, 28, 28)
        return torch.from_numpy(np.array(images)), torch.from_numpy(np.array(labels))

class InceptionV1(nn.Module):
    def __init__(self, in_c, out_c1, out_c2, out_c3, out_c4):
        super().__init__()
        self.bran_1 = nn.Sequential(
            nn.Conv2d(in_c, out_c1, 1, 1, 0),
            nn.ReLU(inplace=True),
        )
        self.bran_2 = nn.Sequential(
            nn.Conv2d(in_c, out_c2[0], 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c2[0], out_c2[1], 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.bran_3 = nn.Sequential(
            nn.Conv2d(in_c, out_c3[0], 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c3[0], out_c3[1], 5, 1, 2),
            nn.ReLU(inplace=True),
        )
        self.bran_4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_c, out_c4, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.bran_1(x)
        x2 = self.bran_2(x)
        x3 = self.bran_3(x)
        x4 = self.bran_4(x)
        return torch.cat((x1, x2, x3, x4), dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer3 = nn.Sequential(
            InceptionV1(192, 64, [96, 128], [16, 32], 32),
            InceptionV1(256, 128, [128, 192], [32, 96], 64),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer4 = nn.Sequential(
            InceptionV1(480, 192, [96, 208], [16, 48], 64),
            InceptionV1(512, 160, [112, 224], [24, 64], 64),
            InceptionV1(512, 128, [128, 256], [24, 64], 64),
            InceptionV1(512, 112, [144, 288], [32, 64], 64),
            InceptionV1(528, 256, [160, 320], [32, 128], 128),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer5 = nn.Sequential(
            InceptionV1(832, 256, [160, 320], [32, 128], 128),
            InceptionV1(832, 384, [192, 384], [48, 128], 128),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc(x)
        return x

model = GoogLeNet(10).to(device)

loss_iter, acc_iter = [], []

trainset = FashionMnistDataset(root_dir, True)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=16)

testset = FashionMnistDataset(root_dir, False)
testloader = DataLoader(dataset=testset, batch_size=batch_size, num_workers=16)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
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
            x, y = x.to(device), y.to(device)
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
    train(trainloader, model, criterion, optimizer)
    test(testloader, model, criterion)
    if (epoch+1) % 10 == 0:
        save_checkpoint('model.pth', model, optimizer)

plot_loss(loss_iter, acc_iter)
