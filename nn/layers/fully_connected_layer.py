import numpy as np

from nn.activations import Activation, LogisticActivation
from nn.layers import Layer
from nn.types import require_type


class FullyConnectedLayer(Layer):

    def __init__(self, size, activation=LogisticActivation(), weights="normal", scale=(1, 0)):
        super(FullyConnectedLayer, self).__init__(size)

        require_type(activation, Activation)
        self.activation = activation

        self.w = np.zeros((0, 0))
        self.b = np.zeros(0)
        self.weights = weights
        self.scale = scale

        self.z = np.zeros(size)
        self.a = np.zeros(size)
        self.d = np.zeros(size)
        self.delta = np.zeros(size)

    def connect(self, prev_layer):
        require_type(prev_layer, Layer)
        if prev_layer is None:
            raise ValueError("Needs previous layer to connect")

        if self.weights == "normal":
            self.w = np.random.standard_normal((self.size, prev_layer.size))
            self.b = np.random.standard_normal(self.size)
        elif self.weights == "uniform":
            self.w = np.random.random((self.size, prev_layer.size))
            self.b = np.random.random(self.size)
        elif self.weights == "zeros":
            self.w = np.zeros((self.size, prev_layer.size))
            self.b = np.zeros(self.size)
        else:
            raise ValueError("weights should be 'normal' / 'uniform' / 'zeros'")

        (scale_zoom, scale_bias) = self.scale
        self.w = self.w * scale_zoom - scale_bias
        self.b = self.b * scale_zoom - scale_bias

    def apply(self, x):
        self.z = self.w.dot(x) + self.b
        self.a = self.activation.apply(self.z)
        return self.a

    # def calc_prev_error(self, delta):
    #     return self.w.transpose().dot(self.delta)
    #
    # def calc_error(self, next_layer, target=None):
    #     if next_layer is None:
    #         self.delta = self.a - target
    #     else:
    #         self.delta = next_layer.calc_prev_error() * self.activation.apply_derivative(self.z)
    #     return self.delta
    #
    # def calc_dcost(self, next_layer):
    #     self.d += next_layer.delta.dot(self.a.transpose())
