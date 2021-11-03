import os
import gzip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from torch.utils.data import Dataset, DataLoader

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

pre_train = False
root_dir = os.path.expanduser('~/dataset/fashion_mnist')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model params from paper
model_params = {
    'conv1': [1, 96, 11, 4, 0],
    'max1': [3, 2, 0],
    'conv2': [96, 256, 5, 1, 2],
    'max2': [3, 2, 0],
    'conv3': [256, 384, 3, 1, 1],
    'conv4': [384, 384, 3, 1, 1],
    'conv5': [384, 256, 3, 1, 1],
    'max3': [3, 2, 0],
    'fc1': [256*5*5, 4096],
    'fc2': [4096, 4096],
    'fc3': [4096, 10],
}

# model params for single gpu
# model_params = {
    # 'conv1': [1, 48, 11, 4, 0],
    # 'max1': [3, 2, 0],
    # 'conv2': [48, 128, 5, 1, 2],
    # 'max2': [3, 2, 0],
    # 'conv3': [128, 192, 3, 1, 1],
    # 'conv4': [192, 192, 3, 1, 1],
    # 'conv5': [192, 128, 3, 1, 1],
    # 'max3': [3, 2, 0],
    # 'fc1': [128*5*5, 2048],
    # 'fc2': [2048, 2048],
    # 'fc3': [2048, 10],
# }

# model params from torch
# model_params = {
    # 'conv1': [1, 64, 11, 4, 2],
    # 'max1': [3, 2, 0],
    # 'conv2': [64, 192, 5, 1, 2],
    # 'max2': [3, 2, 0],
    # 'conv3': [192, 384, 3, 1, 1],
    # 'conv4': [384, 256, 3, 1, 1],
    # 'conv5': [256, 256, 3, 1, 1],
    # 'max3': [3, 2, 0],
    # 'fc1': [256*6*6, 4096],
    # 'fc2': [4096, 4096],
    # 'fc3': [4096, 10],
# }

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
        data = VF.resize(data, (224, 224))
        return data.div(255).float(), target.long()

    def __len__(self):
        return len(self._target)

    def _read_data(self, image_file, label_file):
        with gzip.open(label_file, 'rb') as f:
            labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        with gzip.open(image_file, 'rb') as f:
            images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 1, 28, 28)
        return torch.from_numpy(images), torch.from_numpy(labels)

class AlexNet(nn.Module):
    def __init__(self, params, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(*params['conv1']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(*params['max1']),
            nn.Conv2d(*params['conv2']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(*params['max2']),
            nn.Conv2d(*params['conv3']),
            nn.ReLU(inplace=True),
            nn.Conv2d(*params['conv4']),
            nn.ReLU(inplace=True),
            nn.Conv2d(*params['conv5']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(*params['max3']),
        )
        self.fc = nn.Sequential(
            nn.Linear(*params['fc1']),
            nn.Dropout(0.5),
            nn.Linear(*params['fc2']),
            nn.Dropout(0.5),
            nn.Linear(*params['fc3']),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = AlexNet(model_params, 10).to(device)

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
