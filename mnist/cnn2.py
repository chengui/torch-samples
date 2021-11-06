import os
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import sys
sys.path.insert(0, os.path.abspath('../basic'))
from classification import train

num_epochs = 10
batch_size = 64
learning_rate = 1e-2

net = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(20*4*4, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

root_dir = os.path.expanduser('~/dataset/mnist')

trainset = MNIST(root=root_dir, train=True, download=True, transform=ToTensor())
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=8)

testset = MNIST(root=root_dir, train=False, download=True, transform=ToTensor())
testloader = DataLoader(dataset=testset, batch_size=batch_size, num_workers=8)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

train(net, trainloader, testloader, loss_fn, optimizer)
