import numpy as np

from nn.activations import Activation, LogisticActivation
from nn.layers import Layer
from nn.types import require_type


class FullyConnectedLayer(Layer):
    def __init__(self, in_len, out_len, activation=LogisticActivation(), weights="uniform", scale=(1, 0)):
        self.in_len = in_len
        self.out_len = out_len

        require_type(activation, Activation)
        self.activation = activation

        if weights == "zeros":
            self.w = np.zeros((out_len, in_len))
        elif weights == "uniform":
            (scale_zoom, scale_bias) = scale
            self.w = (np.random.random((out_len, in_len)) * 2 - 1) * scale_zoom - scale_bias
        else:
            self.w = np.zeros((0, 0))  # To force type
            raise ValueError("weights should be 'zeros' or 'uniform'")

        self.z = np.zeros(out_len)
        self.a = np.zeros(out_len)
        self.delta = np.zeros(out_len)

    def apply(self, vector):
        self.z = self.w.dot(vector)
        self.a = self.activation.apply(self.z)
        return self.a

    def calculate_prev_layer_error(self, delta):
        return self.w.transpose().dot(delta)

    def calculate_error(self, delta):
        self.delta += (delta * self.activation.apply_derivative(self.z))

    def require_connect(self, prev_layer):
        if isinstance(prev_layer, FullyConnectedLayer):
            if prev_layer.out_len != self.in_len:
                raise ValueError("Previous layer outputs({0}) don't match this layer inputs({1})".
                                 format(prev_layer.out_len, self.in_len))
        else:
            raise ValueError("Expected previous layer to be {0}".format(self.__class__))

