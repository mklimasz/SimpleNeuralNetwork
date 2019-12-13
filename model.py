from typing import List

from layers import Layer
from loss import Loss
from optimizers import Optimizer


class Model:

    def __init__(self, layers: List[Layer], optimizer: Optimizer, criterion: Loss):
        self.layers = layers
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, x, y):
        for idx, layer in enumerate(self.layers):
            self.optimizer.add_input(idx, x)
            x = layer.forward(x)

        logits = x
        loss, pred = self.criterion.forward(y, logits)
        return loss, pred, logits

    def backward(self, y, logits):
        upstream_grad = self.criterion.backward(y, logits)
        for idx, layer in reversed(list(enumerate(self.layers))):
            upstream_grad, grad = layer.backward(self.optimizer.get_input(idx), upstream_grad)
            self.optimizer.update_layer(grad, layer)
