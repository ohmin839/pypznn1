import numpy as np
from pypznn1.deeplearning import Function
from pypznn1.deeplearning.core import as_variable
from pypznn1.deeplearning import utils

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = np.exp(x) * gy
        return gx

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            gx = transpose(gy)
            return gx
            
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        gx = transpose(gy)
        return gx

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def sin(x):
    f = Sin()
    return f(x)

def cos(x):
    f = Cos()
    return f(x)

def tanh(x):
    f = Tanh()
    return f(x)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    f = Reshape(shape)
    return f(x)

def transpose(x, axes=None):
    f = Transpose(axes)
    return f(x)

def sum(x, axis=None, keepdims=False):
    f = Sum(axis, keepdims)
    return f(x)
