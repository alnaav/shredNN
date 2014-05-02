from nn.train.trainer import Trainer
import numpy as np


class LayerData:
    def __init__(self, layer):
        self.z = np.zeros(layer.size)
        self.a = np.zeros(layer.size)
        self.neuron_error = np.zeros(layer.size)
        self.grad_w = np.zeros(layer.w.shape)
        self.grad_b = np.zeros(layer.b.shape)
        self.d_w = np.zeros(layer.w.shape)
        self.d_b = np.zeros(layer.b.shape)


class GradientDescentTrainer(Trainer):
    def __init__(self, regularization_param=0.01, learning_rate=0.2):
        self.iteration_number = 10000
        self.l = regularization_param
        self.a = learning_rate
        self.__set_coeff__(1)

    def __set_coeff__(self, samples_len):
        self.rev_m = 1.0 / samples_len
        self.coeff = self.l * self.rev_m * 0.5

    def calc_gradient(self, curr, a_prev, w):
        curr.grad_w += curr.neuron_error.transpose().dot(a_prev)
        curr.grad_b += curr.neuron_error[1, :]
        curr.d_w = self.rev_m * curr.grad_w + self.l * w
        curr.d_b = self.rev_m * curr.grad_b

    def step(self, layers, x, y):
        layers_data = [LayerData(layer) for layer in layers[1:]]

        a = x
        for i, layer in enumerate(layers[1:]):
            layers_data[i].z = a.dot(layer.w.transpose())
            layers_data[i].z += layer.b
            layers_data[i].a = layer.activation.apply(layers_data[i].z)
            a = layers_data[i].a

        cost = self.cost(layers, layers_data[-1].a, y)

        curr = layers_data[-1]
        curr.neuron_error = curr.a - y
        for i in range(len(layers) - 1, 1, -1):
            prev = layers_data[i - 2]
            curr = layers_data[i - 1]

            prev.neuron_error = curr.neuron_error.dot(layers[i].w) * layers[i - 1].activation.apply_derivative(prev.z)
            self.calc_gradient(curr, prev.a, layers[i].w)

        self.calc_gradient(layers_data[0], x, layers[1].w)

        for layer, data in zip(layers[1:], layers_data):
            layer.w -= self.a * data.d_w
            layer.b -= self.a * data.d_b

        return cost

    def cost(self, layers, predicted, expected):
        hv = predicted.ravel()
        yv = expected.ravel()

        reg = 0
        for layer in layers[1:]:
            reg += np.sum(layer.w * layer.w)
        reg *= self.coeff

        err = -(np.log2(hv).transpose().dot(yv) + np.log2(1 - hv).transpose().dot((1 - yv))) * self.rev_m
        return err + reg

    def train(self, nn, features, target, k):
        self.__set_coeff__(features.shape[0])

        y = np.zeros((target.shape[0], k))
        for i in range(0, y.shape[0]):
            y[i, target[i]] = 1

        cost = 100
        for i in range(0, self.iteration_number):
            cost = self.step(nn.layers, features, y)
        print cost



