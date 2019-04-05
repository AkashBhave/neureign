import numpy as np

from neureign import activations


class ANN:
    def __init__(self):
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3

        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1);
        self.z2 = activations.sigmoid(self.z1)
        self.z3 = np.dot(self.z2, self.w2)

        output = activations.sigmoid(self.z3)
        return output
