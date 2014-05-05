import unittest
from numpy.testing import assert_allclose
import numpy as np

from gtnn.generators.mlp import mlp
from gtnn.learn.momentum import train
from gtnn.network.activation import LogSigmoid


class TestMomentum(unittest.TestCase):

    def test_simpleChain(self):
        inp = [[1]]
        out = [[0.412]]
        n = mlp(sizes=[1, 1, 1],
                weightGenerator=np.random.random,
                biasGenerator=np.random.random,
                activationFunction=LogSigmoid(1, -1))

        train(net=n, inputs=inp, outputs=out,
              numEpochs=100, learningRate=0.8)

        assert_allclose([n.forward(i) for i in inp], out, atol=0.1)

    def test_simpleFork(self):
        inp = [[1]]
        out = [[0.412, 0.9]]
        n = mlp(sizes=[1, 2],
                weightGenerator=np.random.random,
                biasGenerator=np.random.random,
                activationFunction=LogSigmoid(-1, 1))

        train(net=n, inputs=inp, outputs=out,
              numEpochs=100, learningRate=0.8)

        assert_allclose([n.forward(i) for i in inp], out, atol=0.1)

    def test_xor(self):
        inp = [[0, 0], [1, 0], [0, 1], [1, 1]]
        out = [[0], [1], [1], [0]]

        n = mlp(sizes=[2, 2, 1],
                weightGenerator=np.random.random,
                biasGenerator=np.random.random,
                activationFunction=LogSigmoid(0, 1))

        train(net=n, inputs=inp, outputs=out,
              numEpochs=1000, learningRate=0.3, momentum=0.8)

        assert_allclose([n.forward(i) for i in inp], out, atol=0.1)
