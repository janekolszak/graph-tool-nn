import unittest
from numpy.testing import assert_allclose

from gtnn.generators.mlp import *
from gtnn.network.activation import Identity


class TestMultilayerPerceptron(unittest.TestCase):

    def test_construct(self):
        n = mlp([1, 2, 2, 1])
        n.forward([1])

    def test_forwardChain(self):
        n = mlp(sizes=[1, 1, 1],
                weightGenerator=lambda: 1,
                biasGenerator=lambda: 0,
                activationFunction=Identity())
        assert_allclose(n.forward([0]), [0])
        assert_allclose(n.forward([10]), [10])
        assert_allclose(n.forward([-11]), [-11])

    def test_forwardFork12(self):
        n = mlp(sizes=[1, 2],
                weightGenerator=lambda: 1,
                biasGenerator=lambda: 0,
                activationFunction=Identity())
        assert_allclose(n.forward([0]), [0, 0])
        assert_allclose(n.forward([10]), [10, 10])
        assert_allclose(n.forward([-11]), [-11, -11])

    def test_forwardFork21(self):
        n = mlp(sizes=[2, 1],
                weightGenerator=lambda: 1,
                biasGenerator=lambda: 0,
                activationFunction=Identity())
        assert_allclose(n.forward([0, 0]), [0])
        assert_allclose(n.forward([10, 10]), [20])
        assert_allclose(n.forward([-11, -10]), [-21])
