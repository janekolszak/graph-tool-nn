import unittest
from numpy.testing import assert_allclose

import graph_tool.all as gt
import numpy as np

from gtnn.generators.multilayer_perceptron import multilayer_perceptron
from gtnn.learn.simple import train
from gtnn.network.activation import LogSigmoid, Identity
from gtnn.network.net import Net


class TestSimple(unittest.TestCase):

    def test_simpleChain(self):
        inp = [[1]]
        out = [[0.412]]
        n = multilayer_perceptron(sizes=[1, 1, 1],
                                  weightGenerator=np.random.random,
                                  biasGenerator=np.random.random,
                                  activationFunction=LogSigmoid(1, -1))

        train(net=n, inputs=inp, outputs=out,
              numEpochs=200, learningRate=0.8)
        assert_allclose([n.forward(i) for i in inp], out, atol=0.1)

    def test_simpleFork(self):
        inp = [[1]]
        out = [[0.44, 0.77, 0.33]]
        n = multilayer_perceptron(sizes=[len(inp[0]), len(out[0])],
                                  weightGenerator=np.random.random,
                                  biasGenerator=np.random.random,
                                  activationFunction=LogSigmoid(0,1))

        train(net=n, inputs=inp, outputs=out,
              numEpochs=100, learningRate=0.7)
        assert_allclose([n.forward(i) for i in inp], out, atol=0.1)


    # def test_simpleFork2(self):
    #     inp = [[0, 0], [1,0],[1,1],[0,1]]
    #     out = [[0], [1], [0], [1]]

    #     n = multilayer_perceptron(sizes=[2, 3, 1],
    #                               weightGenerator=lambda: 0,
    #                               biasGenerator=lambda: 0,
    #                               activationFunction=LogSigmoid(-1, 1))
    #     # print(n)
    #     train(net=n, inputs=inp, outputs=out,
    #           numEpochs=10000, learningRate=0.3)
    #     # print(n)

    #     assert_allclose([n.forward(i) for i in inp], out, atol=0.1)

    # def test_xor(self):
    #     inp = [[0, 0], [1, 0], [0, 1], [1, 1]]
    #     out = [[0], [1], [1], [0]]

    #     n = multilayer_perceptron(sizes=[2, 2, 1],
    #                               weightGenerator=lambda: 0,
    #                               biasGenerator=lambda: 0,
    #                               activationFunction=LogSigmoid(0, 1))
    #     # print(n)
    #     train(net=n, inputs=inp, outputs=out,
    #           numEpochs=10000, learningRate=0.3)
    #     # print(n)

    #     assert_allclose([n.forward(i) for i in inp], out, atol=0.1)
