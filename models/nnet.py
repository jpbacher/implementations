import numpy as np


class BaseLayer:
    """Base class for all layers. Contains methods for connecting layers."""
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
        self.batch_norm_p = batch_norm_p
        self.mu_h = 0
        self.std_h= 1

    def forward(self, z_in):
        h = z_in @ self.w + self.b
        mu_h = np.mean(h, keepdims=True, axis=0)
        self.mu_h = self.mu_h * self.batch_norm_p + mu_h * (1 - self.batch_norm_p)
        std_h = np.std(h, keepdims=True, axis=0)
        self.std_h = self.std_h * self.batch_norm_p + std_h * (1 - self.batch_norm_p)
        h_norm = (h - self.mu_h) / self.std_h
        z_out = self.activation(h_norm)
        return z_out

    def backward(self, z_in, y):
        h = z_in @ self.w + self.b
        mu_h = np.mean(h, keepdims=True, axis=0)
        self.mu_h = self.mu_h * self.batch_norm_p + mu_h * (1 - self.batch_norm_p)
        std_h = np.std(h, keepdims=True, axis=0)
        self.std_h = self.std_h * self.batch_norm_p + std_h * (1 - self.batch_norm_p)
        h_norm = (h - self.mu_h) / self.std_h
        z_out = self.activation(h_norm)

        grad_after = self.after.backward(z_out, y)
        grad_h_norm = grad_after * self.activation.deriv(h_norm)
        grad_h = grad_h_norm / self.std_h

        grad_w = z_in.T @ grad_h
        grad_b = np.sum(grad_h, axis=0)
        grad_z_in = grad_h @ self.w.T

        self.w = grad_w
        self.b =grad_b

        return grad_z_in

    def predict(self, z_in):
        return self.after.predict(self.forward(z_in))

    def get_params(self):
        params = self.after.get_params()
        my_params = [self.w, self.b]
        params.extend(my_params)
        return params


class OutputLayer(Layer):

    def backward(self, z_in, y):
        y_hat = self.forward(z_in)
        grad_h = y_hat - y

        grad_w = z_in.T @ grad_h
        grad_b = np.sum(grad_h, axis=0)
        grad_z_in = grad_h @ self.w.T
        return grad_z_in

    def predict(self, z_in):
        return self.forward(z_in)

    def get_params(self):
        my_params = [self.w, self.b]
        return my_params


class EmptyLayer(BaseLayer):
    def __init__(self, grad):
        super().__init__()
        self.grad = grad

    def backward(self, **args, **kwargs):
        return self.grad

    @staticmethod
    def forward(z_in):
        return z_in

    @staticmethod
    def predict(z_in):
        return z_in

    @staticmethod
    def get_params():
        return []


class SplitLayer(BaseLayer):

    def __init__(self, left, right):
        super(SplitLayer, self).__init__()
        self.left = left
        self.right = right
        self.left.add_after(EmptyLayer(0))
        self.right.add_after(EmptyLayer(0))

    def forward(self, z_in):
        z_left = self.left.forward(z_in)
        z_right - self.right.forward(z_in)
        z_out = np.hstack([z_left, z_right])
        return z_out



