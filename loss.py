from typing import Tuple

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


class Loss:

    def forward(self, y_true: np.ndarray, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns loss value and prediction."""
        pass

    def backward(self, y_true: np.ndarray, logits: np.ndarray):
        pass


class CrossEntropy(Loss):

    def forward(self, y_true, logits):
        prob = softmax(logits)
        return - np.log(prob[range(logits.shape[0]), y_true]), np.argmax(prob, axis=1)

    def backward(self, y_true, logits):
        grad = softmax(logits)
        grad[range(logits.shape[0]), y_true] -= 1
        return grad
