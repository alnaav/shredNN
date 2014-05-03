from nn.layers import Layer
from nn.layers.id_layer import IdLayer
from nn.types import require_type
from nn.train.gradient_descent_trainer import GradientDescentTrainer


class NN(object):

    def __init__(self, input_size):
        self.input_size = input_size

        self.layers = []
        self.layers.append(IdLayer(input_size))

    def add_layer(self, layer):
        require_type(layer, Layer)

        layer.connect(self.layers[-1])
        self.layers.append(layer)

    # def apply(self, vector):
    #     if vector.size != self.input_size:
    #         raise ValueError("Invalid vector size - expected: ({0}), got: ({1})".format(
    #             self.input_size
    #         ))
    #
    #     curr = vector
    #     for layer in self.layers:
    #         curr = layer.apply(curr)
    #     return curr

    def apply(self, features):
        if features.shape[1] != self.input_size:
            raise ValueError("Invalid vector size - expected: ({0}), got: ({1})".format(
                self.input_size
            ))

        curr = features
        for layer in self.layers:
            curr = layer.apply(curr)
        return curr

    def train(self, features, target):
        trainer = GradientDescentTrainer()
        trainer.train(self, features, target, 10)


