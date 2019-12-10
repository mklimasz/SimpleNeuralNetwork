class Optimizer:

    def __init__(self):
        self.cache = {}

    def get_input(self, idx):
        return self.cache[idx]

    def add_input(self, idx, x):
        self.cache[idx] = x

    def weights_update(self, grad, layer):
        weights = layer.weights()
        assert len(weights) == len(grad)
        updates = []
        for weight, grad in zip(weights, grad):
            update = self.step(grad, weight)
            updates.append(update)
        return updates

    def zero_grad(self):
        self.cache = {}

    def step(self, grad, weight):
        pass


class SGD(Optimizer):

    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def step(self, grad, weight):
        return weight - self.learning_rate * grad.mean(axis=0)
