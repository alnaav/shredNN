from nn.activations.activation import Activation
import numpy as np


class IdActivation(Activation):
    def apply(self, vector):
        return vector

    def apply_derivative(self, vector):
        return np.ones(vector.shape, dtype=int)
