from typing import Tuple, List, Dict, Any

import numpy as np


class Param:

    def __init__(self, data: np.ndarray):
        self.data = data


class Layer:

    def __init__(self, train: bool = True):
        """Creates layer.

        :param train: bool deciding whether the layer is in train/eval mode
        """
        self.train = train

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Any], Dict[str, Any]]:
        """Forward pass.

        :param x: input of the layer
        :return: output of the layer, *args and **kwargs as a tuple
        Args and kwargs are passed as arguments for backward pass.
        """
        pass

    def backward(self, x: np.ndarray, dy: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Backward pass.

        :param x: the layer input
        :param dy: upstream gradient
        :param args: optional args
        :param kwargs: optional kwargs
        :return: tuple of downstream gradient, then list of gradients with respect to parameters (in order)
        defined in weights method
        """
        pass

    def weights(self) -> List[Param]:
        """Learnable parameters of the layer - order must be the same as in backward."""
        return []


def xavier_uniform_init(input_dim, output_dim, gain: float = 1.0):
    r = gain * np.sqrt(6.0 / (input_dim + output_dim))
    return np.random.uniform(-r, r, (input_dim, output_dim))


class LinearLayer(Layer):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = Param(xavier_uniform_init(input_dim, output_dim))
        self.b = Param(np.zeros(shape=(1, output_dim)))
        self.output_dim = output_dim

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Any], Dict[str, Any]]:
        y = (np.matmul(x.transpose((0, 2, 1)), self.W.data) + self.b.data).transpose((0, 2, 1))
        assert y.shape == (x.shape[0], self.output_dim, 1)
        return y, [], {}

    def backward(self, x: np.ndarray, dy: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        batch_size = x.shape[0]

        dx = dy.transpose((0, 2, 1)).dot(self.W.data.T).transpose((0, 2, 1))
        assert dx.shape == x.shape

        dW = np.matmul(dy, x.transpose((0, 2, 1))).transpose((0, 2, 1))
        assert dW.shape == (batch_size, *self.W.data.shape)

        db = dy.transpose((0, 2, 1))
        assert db.shape == (batch_size, *self.b.data.shape)

        return dx, [dW, db]

    def weights(self) -> List[Param]:
        return [self.W, self.b]


class ReLU(Layer):

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Any], Dict[str, Any]]:
        return np.maximum(x, 0), [], {}

    def backward(self, x: np.ndarray, dy: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        dx = np.maximum(x, 0) * dy
        assert dy.shape == dx.shape
        return dx, []


class Dropout(Layer):

    def __init__(self, keep_rate: float):
        super().__init__()
        self.keep_rate = keep_rate

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Any], Dict[str, Any]]:
        mask = (np.random.binomial(1, self.keep_rate, size=x.shape) / (1 - self.keep_rate)
                if self.train else np.ones_like(x))
        return mask * x, [], {"mask": mask}

    def backward(self, x: np.ndarray, dy: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        assert "mask" in kwargs
        return kwargs["mask"] * dy, []
