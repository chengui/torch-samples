import glob
import torch
import matplotlib.pyplot as plt


def load_checkpoint(model, optimizer, filename):
    files = glob.glob(f'{filename}-[0-9]*')
    if len(files) == 0: return 0
    files.sort(key=lambda x: int(x.split('-')[-1]))
    print(f'Load model parameters from {files[-1]}')
    checkpoint = torch.load(files[-1])
    model.load_state_dict(checkpoint['state'])
    optimizer.load_state_dict(checkpoint['optim'])
    return int(files[-1].split('-')[-1])

def save_checkpoint(model, optimizer, filename, epoch):
    checkpoint = {
        'state': model.state_dict(),
        'optim': optimizer.state_dict(),
    }
    savefile = f'{filename}-{epoch}'
    print(f'Save model parameters to {savefile}')
    torch.save(checkpoint, savefile)

def train_epoch(net, dataloader, loss_fn, optimizer, device):
    net.train()
    train_loss, train_acc = 0., 0.
    num_batches, num_samples = len(dataloader), len(dataloader.dataset)
    batch_size, batch_print = dataloader.batch_size, num_batches // 5
    for idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        train_acc += (torch.argmax(y_hat.data, dim=1)==y).float().sum()
        if (idx+1) % batch_print == 0 or (idx+1) == num_batches:
            loss, curr = loss.item(), idx*dataloader.batch_size+len(x)
            print(f'loss={loss:>7f} [{curr:>5d}/{num_samples:>5d}]')
    train_loss /= num_batches
    train_acc /= num_samples
    return train_loss, train_acc

def test_epoch(net, dataloader, loss_fn, optimizer, device):
    net.eval()
    test_loss, test_acc = 0., 0.
    num_batches, num_samples = len(dataloader), len(dataloader.dataset)
    with torch.no_grad():
        for (x, y) in dataloader:
            y_hat = net(x)
            loss = loss_fn(y_hat, y)
            test_loss += loss.item()
            test_acc += (y_hat.argmax(1)==y).float().sum()
    test_loss /= num_batches
    test_acc /= num_samples
    return test_loss, test_acc

def plot(iter_loss, iter_acc, fname='epoch.png'):
    plt.subplot(211)
    plt.ylabel('Loss')
    if isinstance(iter_loss[0], tuple):
        plt.plot(range(len(iter_loss)), [i[0] for i in iter_loss], linestyle='-', label='train loss')
        plt.plot(range(len(iter_loss)), [i[1] for i in iter_loss], linestyle='--', label='test loss')
    else:
        plt.plot(range(len(iter_loss)), iter_loss, linestyle='-', label='train loss')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    if isinstance(iter_acc[0], tuple):
        plt.plot(range(len(iter_acc)), [i[0] for i in iter_acc], linestyle='-', label='train acc')
        plt.plot(range(len(iter_acc)), [i[1] for i in iter_acc], linestyle='--', label='test acc')
    else:
        plt.plot(range(len(iter_acc)), iter_acc, linestyle='-', label='train loss')
    plt.legend(loc='lower right')
    plt.savefig(fname)

def train(net, trainloader, testloader, loss_fn, optimizer, checkpoint=10, modelname='model.pth', device=None, num_epochs=10):
    if not device:
        device = next(net.parameters()).device
    start_epoch = 0
    if checkpoint > 0:
        start_epoch = load_checkpoint(net, optimizer, modelname)
    print(f'Start to train on device {device}')
    iter_loss, iter_acc = [], []
    for epoch in range(start_epoch, start_epoch+num_epochs):
        print(f'Epoch {epoch} {"":->30s}')
        train_loss, train_acc = train_epoch(net, trainloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_epoch(net, testloader, loss_fn, optimizer, device)
        iter_loss.append((train_loss, test_loss))
        iter_acc.append((train_acc, test_acc))
        print(f'Train: loss={train_loss:.3f}, acc={train_acc:.3f}; Test: loss={test_loss:.3f}, acc={test_acc:.3f}')
        if checkpoint > 0 and (epoch+1) % checkpoint == 0:
            save_checkpoint(net, optimizer, modelname, epoch+1)
    plot(iter_loss, iter_acc)
