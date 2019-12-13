import numpy as np

from layers import Layer, Param


class Optimizer:

    def __init__(self):
        self.cache = {}

    def get_input(self, idx):
        return self.cache[idx]

    def add_input(self, idx, x):
        self.cache[idx] = x

    def update_layer(self, grad: np.ndarray, layer: Layer):
        weights = layer.weights()
        assert len(weights) == len(grad)
        for weight, grad in zip(weights, grad):
            self.step(grad, weight)

    def zero_grad(self):
        self.cache = {}

    def step(self, grad: np.ndarray, weight: Param):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def step(self, grad: np.ndarray, weight: Param):
        weight.data -= self.learning_rate * grad.mean(axis=0)
