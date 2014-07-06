import numpy as np
npa = np.array

class SoftmaxGroup:

class Softmax:
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



def softmax(w, t=1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist

if __name__ == '__main__':

    w = np.array([0.0, 0.0])
    print (softmax(w))

    w = np.array([-0.1, 0.2])
    print (softmax(w))

    w = np.array([0.9, -10])
    print (softmax(w))
