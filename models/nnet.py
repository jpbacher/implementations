import numpy as np


class BaseLayer:
    """Base class for all layers. Contains methods for connecting
    layers together.
    """
    def __init__(self):
        self.end = end
        self.after = after

    def add_after(self, layer):
        self.end.after = layer
        self.end = layer
        return self

    def __rshift__(self, layer):
        """Wrapper for self.add_after(). Allows to use >> operator for
        constructing models.
        """
        return self.add_after(layer)


class Layer(BaseLayer):
    def __init__(self, size_in, size_out, activation, lr, he=False, batch_norm_p=1)
        super().__init__()
        if he:
            self.w = np.random.randn((size_in, size_out) * np.sqrt(1 / size_in*size_out))
            self.b = np.random.randn((1, size_out) * np.sqrt(1 / size_in*size_out))
        else:
            self.w = np.random.randn(size_in, size_out)
            self.b = np.random.randn(1, size_out)
        self.activation = activation
        self.lr = lr
