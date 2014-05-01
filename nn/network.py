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
        predicted = self.apply(X)

        self.layers[-1].calc_error(None, y)
        for i, layer in reversed(enumerate(self.layers[:-1])):
            layer.calc_error(self.layers[i-1])
            layer.calc_dcost(self.layers[i-1])


