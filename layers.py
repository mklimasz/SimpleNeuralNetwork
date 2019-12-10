from typing import Tuple, List

import numpy as np


class Layer:

    def forward(self, x: np.array) -> np.array:
        """Forward pass."""
        pass

    def backward(self, x, dy) -> Tuple[np.array, List[np.array]]:
        """Backward pass.

        :param x: layer input
        :param dy: upstream gradient
        :return: tuple of downstream gradient, then list of gradients with respect to parameters (in order)
        defined in weights method
        """
        pass

    def weights(self) -> List[np.array]:
        """Learnable parameters of the layer - order must be the same as in backward."""
        return []

    def update_weights(self, updated_weights: List[np.array]):
        """Replace all weights with theirs new values."""
        pass


class LinearLayer(Layer):

    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(*(input_dim, output_dim))
        self.b = np.random.randn(*(1, output_dim))
        self.output_dim = output_dim

    def forward(self, x):
        y = (np.matmul(x.transpose(0, 2, 1), self.W) + self.b).transpose(0, 2, 1)
        assert y.shape == (x.shape[0], self.output_dim, 1)
        return y

    def backward(self, x, dy):
        batch_size = x.shape[0]

        dx = dy.transpose(0, 2, 1).dot(self.W.T).transpose(0, 2, 1)
        assert dx.shape == x.shape

        dW = np.matmul(dy, x.transpose(0, 2, 1)).transpose(0, 2, 1)
        assert dW.shape == (batch_size, *self.W.shape)

        db = dy.transpose(0, 2, 1)
        assert db.shape == (batch_size, *self.b.shape)

        return dx, [dW, db]

    def weights(self):
        return [self.W, self.b]

    def update_weights(self, updated_weights):
        assert len(updated_weights) == 2
        assert self.W.shape == updated_weights[0].shape
        assert self.b.shape == updated_weights[1].shape
        self.W = updated_weights[0]
        self.b = updated_weights[1]


class ReLU(Layer):

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x, dy):
        dx = np.maximum(x, 0) * dy
        assert dy.shape == dx.shape
        return dx, []
