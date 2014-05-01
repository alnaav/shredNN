from nn.layers import Layer
from nn.layers.id_layer import IdLayer
from nn.types import require_type


class NN(object):

    def __init__(self, input_size):
        self.input_size = input_size

        self.layers = []
        self.layers.append(IdLayer(input_size))

    def add_layer(self, layer):
        require_type(layer, Layer)

        layer.connect(self.layers[-1])
        self.layers.append(layer)

    def apply(self, vector):
        if vector.size != self.input_size:
            raise ValueError("Invalid vector size - expected: ({0}), got: ({1})".format(
                self.input_size
            ))

        curr = vector
        for layer in self.layers:
            curr = layer.apply(curr)
        return curr


    # def fit(self, X, y):
    #     predicted = self.apply(X)
    #
    #     self.layers[-1].calc_error(None, y)
    #     for i, layer in reversed(enumerate(self.layers[:-1])):
    #         layer.calc_error(self.layers[i-1])
    #         layer.calc_dcost(self.layers[i-1])


