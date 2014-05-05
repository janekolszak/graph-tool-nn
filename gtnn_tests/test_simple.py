import unittest
from numpy.testing import assert_allclose

import numpy as np

from gtnn.generators.mlp import mlp
from gtnn.learn.simple import train
from gtnn.network.activation import LogSigmoid


class TestSimple(unittest.TestCase):

    def test_simpleChain(self):
        inp = [[1]]
        out = [[0.412]]
        n = mlp(sizes=[1, 1, 1],
                weightGenerator=np.random.random,
                biasGenerator=np.random.random,
                activationFunction=LogSigmoid(1, -1))

        train(net=n, inputs=inp, outputs=out,
              numEpochs=200, learningRate=0.8)
        assert_allclose([n.forward(i) for i in inp], out, atol=0.1)

    def test_simpleFork(self):
        inp = [[1]]
        out = [[0.44, 0.77, 0.33]]
        n = mlp(sizes=[len(inp[0]), len(out[0])],
                weightGenerator=np.random.random,
                biasGenerator=np.random.random,
                activationFunction=LogSigmoid(0, 1))

        train(net=n, inputs=inp, outputs=out,
              numEpochs=100, learningRate=0.7)

        assert_allclose([n.forward(i) for i in inp], out, atol=0.1)


    # def test_xor(self):
    #     inp = [[0, 0], [1, 0], [0, 1], [1, 1]]
    #     out = [[0], [1], [1], [0]]

    #     n = mlp(sizes=[2, 2, 1],
    #             weightGenerator=lambda: 0,
    #             biasGenerator=lambda: 0,
    #             activationFunction=LogSigmoid(0, 1))
    #     train(net=n, inputs=inp, outputs=out,
    #           numEpochs=10000, learningRate=0.3)

    #     assert_allclose([n.forward(i) for i in inp], out, atol=0.1)
