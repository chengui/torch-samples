import os
import torch
from torch.utils.data import Dataset, DataLoader

num_epochs = 100
batch_size = 1

data_file = os.path.join(os.path.expanduser('~/dataset/boston_housing/housing.csv'))

def normalize(data):
    return (data-data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

class HousingDataset(Dataset):
    def __init__(self, datafile='housing.csv'):
        import numpy as np
        np_data = np.loadtxt(datafile, dtype=np.float32)
        self.data = torch.tensor(normalize(np_data[:,[5,10,12,13]]))

    def __getitem__(self, i):
        return self.data[i][:-1], self.data[i][-1]

    def __len__(self):
        return len(self.data)

class LinearModel(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel(3)

loss_iter = []

dataset = HousingDataset(data_file)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

def plot_loss(hist):
    import matplotlib.pyplot as plt
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(len(hist)), hist, label='Train Loss')
    plt.savefig('loss.png')

for epoch in range(num_epochs):
    print(f'Epoch {epoch} ----------')
    train_loss = 0
    for idx, (x, y) in enumerate(dataloader):
        # forward
        y_pred = model(x)
        loss = criterion(y_pred, y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()

        train_loss += loss.item()
        if idx % 100 == 0:
            loss, current = loss.item(), idx * len(x)
            print(f'loss={loss:>7f} [{current:>5d}/{len(dataset):>5d}]')
    train_loss /= len(dataloader)
    loss_iter.append(train_loss)
    print(f'Train Error: loss={train_loss:>7f}')

print(f'Model: {model.state_dict()}')
plot_loss(loss_iter)
