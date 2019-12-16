from typing import List

from layers import Layer
from loss import Loss
from optimizers import Optimizer


class Model:

    def __init__(self, layers: List[Layer], optimizer: Optimizer, criterion: Loss):
        self.layers = layers
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self):
        for l in self.layers:
            l.train = True

    def eval(self):
        for l in self.layers:
            l.train = False

    def forward(self, x, y):
        for idx, layer in enumerate(self.layers):
            new_x, args, kwargs = layer.forward(x)
            self.optimizer.save(idx, (x, args, kwargs))
            x = new_x

        logits = x
        loss, pred = self.criterion.forward(y, logits)
        return loss, pred, logits

    def backward(self, y, logits):
        upstream_grad = self.criterion.backward(y, logits)
        for idx, layer in reversed(list(enumerate(self.layers))):
            x, args, kwargs = self.optimizer.load(idx)
            upstream_grad, grad = layer.backward(x, upstream_grad, *args, **kwargs)
            self.optimizer.update_layer(grad, layer)
