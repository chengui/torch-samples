import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

num_epochs = 10
batch_size = 16
learning_rate = 1e-3

fine_tune = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = os.path.expanduser('~/dataset/hotdog')
norm_params = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
])

trainset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs)
testset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs)

trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(dataset=testset, batch_size=batch_size, num_workers=4)

loss_iter, acc_iter = [], []

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
    test_acc /= len(dataloader.dataset) / 100
    print(f'Test Error: acc={test_acc:>0.3f}, loss={test_loss:>7f}')

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


if fine_tune:
    print('Fine tuning from pretrained resnet18')
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    out = list(map(id, model.fc.parameters()))
    fea = filter(lambda p: id(p) not in out, model.parameters())
    params = [
        {'params': fea},
        {'params': model.fc.parameters(), 'lr': learning_rate*10}
    ]
    optimizer = optim.SGD(params, lr=learning_rate, weight_decay=0.001)
else:
    print('Train resnet18 from scratch')
    model = resnet18(pretrained=False, num_classes=2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
model = model.to(device)

for epoch in range(num_epochs):
    print(f'Epoch {epoch} {"":->60s}')
    start = time.time()
    train(trainloader, model, criterion, optimizer)
    time1 = time.time()
    test(testloader, model, criterion)
    time2 = time.time()
    print(f'Time Cost: train={time1-start:.1f}s, test={time2-time1:.1f}s')

plot_loss(loss_iter, acc_iter)
