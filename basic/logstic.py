import os
import gzip
import torch
from torch.utils.data import Dataset, DataLoader

epochs = 20
batch_size = 1
learn_rate = 1e-2

class MockDataset(Dataset):
    def __init__(self, train=True, num_samples=1000):
        super().__init__()
        self._train = train
        self._mean, self._bias = 1.7, 1.2
        if self._train:
            self._x, self._y = self._generate_data(num_samples, 2)
        else:
            self._x, self._y = self._generate_data(num_samples//10, 2)

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)

    def _generate_data(self, num_samples, in_features):
        ones = torch.ones(num_samples, in_features)
        x0 = torch.normal(self._mean*ones, 1) + self._bias
        y0 = torch.zeros(num_samples, dtype=torch.long)
        x1 = torch.normal(-self._mean*ones, 1) + self._bias
        y1 = torch.ones(num_samples, dtype=torch.long)
        return torch.cat((x0, x1), 0), torch.cat((y0, y1), 0)

class LogisticModel(torch.nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

model = LogisticModel(2, 2)

loss_iter, acc_iter = [], []

trainset = MockDataset(True)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size)

testset = MockDataset(False)
testloader = DataLoader(dataset=testset, batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

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
    acc_iter.append(100*test_acc)
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

for epoch in range(epochs):
    print(f'Epoch {epoch} --------------------')
    train(trainloader, model, criterion, optimizer)
    test(testloader, model, criterion)

print(f'Model: {model.state_dict()}')
plot_loss(loss_iter, acc_iter)
