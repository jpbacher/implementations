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
