"""Module with some basic neuron activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt


class Identity:
    """Identity activation function"""
    def value(self, x):
        """:returns: x"""
        return x

    def derivative(self, x):
        """:returns: always 1.0"""
        return 1.0

# TODO: rename to sigmoid?
class LogSigmoid:
    """ Scaled sigmoid activation function"""
    def __init__(self, outMin, outMax):
        self.outMin = outMin
        self.outMax = outMax
        self.outScale = outMax - outMin

    def value(self, x):
        """:returns: scaled sigmoid"""

        return self.outMin + self.outScale * 1.0 / (1.0 + np.exp(-1 * x))

    def derivative(self, x):
        """:returns: scaled sigmoid's derivative"""
        logSigmoid = 1.0 / (1.0 + np.exp(-1 * x))
        return self.outScale * logSigmoid * (1.0 - logSigmoid)


def main():
    l = LogSigmoid(-10, 10)
    xx = np.linspace(-10, 10, 100)
    yy = [l.value(x) for x in xx]
    dd = [l.derivative(x) for x in xx]
    plt.plot(xx, yy)
    plt.plot(xx, dd)
    plt.show()


if __name__ == "__main__":
    main()
