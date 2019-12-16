import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import layers
import loss
import optimizers
from model import Model


def get_dataset(batch_size):
    train_loader = DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                   ])), shuffle=True, batch_size=batch_size)

    test_loader = DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                   ])), shuffle=True, batch_size=batch_size)
    return train_loader, test_loader


if __name__ == '__main__':
    torch.random.manual_seed(1234)
    np.random.seed(1234)

    epochs = 10
    lr = 0.01
    batch_size = 32

    optimizer = optimizers.SGD(learning_rate=lr)
    criterion = loss.CrossEntropy()
    layers = [
        layers.LinearLayer(784, 512),
        layers.ReLU(),
        layers.Dropout(keep_rate=0.8),
        layers.LinearLayer(512, 512),
        layers.ReLU(),
        layers.Dropout(keep_rate=0.8),
        layers.LinearLayer(512, 10)
    ]
    model = Model(layers, optimizer, criterion)

    train_loader, test_loader = get_dataset(batch_size)
    for epoch_id in range(epochs):
        model.train()
        total = 0
        correct = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.numpy().reshape(y.shape[0], -1, 1)
            y = y.numpy()

            model.optimizer.zero_grad()
            loss, pred, logits = model.forward(x, y)
            model.backward(y, logits)

            correct += np.sum(y == pred.flatten())
            total += y.shape[0]

            if i % 100 == 0:
                print("Loss:", loss.mean())
        print("Accuracy (train) epoch {}: {} %".format(epoch_id + 1, correct / total * 100.0))

        model.eval()
        total = 0
        correct = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.numpy().reshape(y.shape[0], -1, 1)
            y = y.numpy()

            _, pred, _ = model.forward(x, y)

            correct += np.sum(y == pred.flatten())
            total += y.shape[0]

        print("Accuracy (test) epoch {}: {} %".format(epoch_id + 1, correct / total * 100.0))

    total = 0
    correct = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.numpy().reshape(y.shape[0], -1, 1)
        y = y.numpy()

        _, pred, _ = model.forward(x, y)

        correct += np.sum(y == pred.flatten())
        total += y.shape[0]

    print("Accuracy final (train) epoch {}: {} %".format(epochs, correct / total * 100.0))
