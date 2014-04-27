import unittest
from numpy.testing import assert_allclose

import graph_tool.all as gt

from gtnn.generators.multilayer_perceptron import multilayer_perceptron
from gtnn.learn.momentum import train


class TestMomentum(unittest.TestCase):

    def test_simpleChain(self):
        inp = [[1]]
        out = [[0.412]]
        n = multilayer_perceptron([1, 1, 1])

        train(net=n, inputs=inp, outputs=out,
              numEpochs=100, learningRate=0.8)
        assert_allclose([n.forward(i) for i in inp], out)

    def test_simpleFork(self):
        inp = [[1]]
        out = [[0.412, 0.9]]
        n = multilayer_perceptron([1, 2])

        train(net=n, inputs=inp, outputs=out,
              numEpochs=100, learningRate=0.8)
        print(n.forward([1]))
        assert_allclose([n.forward(i) for i in inp], out)

    def test_xor(self):
        inp = [[0, 0], [1, 0], [0, 1], [1, 1]]
        out = [[0], [1], [1], [0]]

        n = multilayer_perceptron([2, 2, 1])
        train(net=n, inputs=inp, outputs=out,
              numEpochs=1000, learningRate=0.3, momentum=0.8)
        assert_allclose([n.forward(i) for i in inp], out)
