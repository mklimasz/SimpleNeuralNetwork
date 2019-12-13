from typing import Tuple, List

import numpy as np


class Param:

    def __init__(self, data: np.ndarray):
        self.data = data


class Layer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        pass

    def backward(self, x: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Backward pass.

        :param x: layer input
        :param dy: upstream gradient
        :return: tuple of downstream gradient, then list of gradients with respect to parameters (in order)
        defined in weights method
        """
        pass

    def weights(self) -> List[Param]:
        """Learnable parameters of the layer - order must be the same as in backward."""
        return []


class LinearLayer(Layer):

    def __init__(self, input_dim, output_dim):
        self.W = Param(np.random.randn(*(input_dim, output_dim)))
        self.b = Param(np.random.randn(*(1, output_dim)))
        self.output_dim = output_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = (np.matmul(x.transpose(0, 2, 1), self.W.data) + self.b.data).transpose(0, 2, 1)
        assert y.shape == (x.shape[0], self.output_dim, 1)
        return y

    def backward(self, x: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        batch_size = x.shape[0]

        dx = dy.transpose(0, 2, 1).dot(self.W.data.T).transpose(0, 2, 1)
        assert dx.shape == x.shape

        dW = np.matmul(dy, x.transpose(0, 2, 1)).transpose(0, 2, 1)
        assert dW.shape == (batch_size, *self.W.data.shape)

        db = dy.transpose(0, 2, 1)
        assert db.shape == (batch_size, *self.b.data.shape)

        return dx, [dW, db]

    def weights(self) -> List[Param]:
        return [self.W, self.b]


class ReLU(Layer):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def backward(self, x: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        dx = np.maximum(x, 0) * dy
        assert dy.shape == dx.shape
        return dx, []
