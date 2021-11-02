import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

batch_size = 1
num_epochs = 10
learning_rate = 0.001

class MockDataset(Dataset):
    def __init__(self, num_samples=1000):
        self._w, self._b = torch.tensor([2.0, -1.2]), torch.tensor([3.1])
        self._x = torch.randn(num_samples, 2, dtype=torch.float32)
        self._y = torch.matmul(self._x, self._w) + self._b

    def __getitem__(self, i):
        return self._x[i], self._y[i]

    def __len__(self):
        return len(self._x)

class LinearModel(torch.nn.Module):
    def __init__(self, in_features=2):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

def plot_loss(hist):
    import matplotlib.pyplot as plt
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(len(hist)), hist, label='Train Loss')
    plt.savefig('loss.png')

model = LinearModel(2)

dataset = MockDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

loss_iter = []
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_loss = []
    print(f'Epoch {epoch} {"":->60s}')
    for idx, (x, y) in enumerate(dataloader):
        # forward
        y_pred = model(x)
        loss = criterion(y_pred, y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()

        train_loss.append(loss.item())
        if idx % 100 == 0:
            loss, current = loss.item(), idx * len(x)
            print(f'loss={loss:>7f} [{current:>5d}/{len(dataset):>5d}]')
    loss_iter.append(sum(train_loss)/len(train_loss))

print(f'Model: {model.state_dict()}')
plot_loss(loss_iter)
