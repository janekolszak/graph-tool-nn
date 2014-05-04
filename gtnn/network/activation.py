import numpy as np
import matplotlib.pyplot as plt


class Identity(object):

    def value(self, x):
        return x

    def derivative(self, x):
        return 1.0


class LogSigmoid(object):

    def __init__(self, outMin, outMax):
        self.outMin = outMin
        self.outMax = outMax
        self.outScale = outMax - outMin

    def value(self, x):
        return self.outMin + self.outScale * 1.0 / (1.0 + np.exp(-1 * x))

    def derivative(self, x):
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
