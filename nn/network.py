from nn.layers import Layer
from nn.types import require_type


class NN(object):

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        require_type(layer, Layer)
        if len(self.layers) != 0:
            layer.require_connect(self.layers[-1])

        self.layers.append(layer)

    def apply(self, vector):
        curr = vector
        for layer in self.layers:
            curr = layer.apply(curr)
        return curr


    def fit(self, X, y):
        delta = [0 in range(0, len(self.layers))]

        a = [X]
        for layer in self.layers:
            a.append(layer.apply(a[-1]))

        delta[-1] = a[-1] - y
        for i, layer in reversed(enumerate(self.layers)):
            d = layer.calculate_prev_error(delta[i])
            delta[i-1] += self.layers[i-1].calculate_error(d)

