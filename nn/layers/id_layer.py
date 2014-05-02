from nn.layers import Layer
from nn.types import require_type


class IdLayer(Layer):

    def __init__(self, size):
        super(IdLayer, self).__init__(size)

    def connect(self, prev_layer):
        require_type(prev_layer, Layer)

        if prev_layer is not None:
            if prev_layer.size != self.size:
                raise ValueError("Layers size mismatch - expected: ({0}), got: ({1})".format(
                    self.size, prev_layer.size
                ))

    def apply(self, x):
        return x
