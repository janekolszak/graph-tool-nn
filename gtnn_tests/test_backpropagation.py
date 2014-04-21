import unittest
from numpy.testing import assert_allclose
import graph_tool.all as gt

from gtnn.learn.backpropagation import BackpropagationNet
from gtnn.network.net import Net
from gtnn.network.activation import Identity


class TestBackpropagation(unittest.TestCase):

    def test_construc(self):
        g = gt.Graph()
        g.add_vertex(4)
        g.add_edge(g.vertex(0), g.vertex(1))
        g.add_edge(g.vertex(0), g.vertex(2))
        g.add_edge(g.vertex(1), g.vertex(3))
        g.add_edge(g.vertex(2), g.vertex(3))
        g.add_vertex(1)
        g.add_edge(g.vertex(4), g.vertex(0))
        n = Net(1, 1, g)
        for e in g.edges():
            n.weightProp[e] = 1
        n.prepare()

        bn = BackpropagationNet(n)

        bn.backward([1])
        bn.forward([1])
        bn.backward([2])

    def test_backwardChain(self):
        g = gt.Graph()
        g.add_vertex(3)
        g.add_edge(g.vertex(0), g.vertex(1))
        g.add_edge(g.vertex(1), g.vertex(2))
        n = Net(1, 1, g)

        bn = BackpropagationNet(n)

        # Init activation functions
        for v in n.g.vertices():
            n.activation[v] = Identity()

        # All weights  == 0
        assert_allclose(bn.errorProp.a, [0, 0, 0])

        bn.backward([1])
        assert_allclose(bn.errorProp.a, [0, 0, 1])

        bn.backward([-100])
        assert_allclose(bn.errorProp.a, [0, 0, -100])

        # Init weights
        for e in n.g.edges():
            n.weightProp[e] = 1.0

        # Tests
        bn.backward([1])
        assert_allclose(bn.errorProp.a, [1, 1, 1])
        bn.backward([30])
        assert_allclose(bn.errorProp.a, [30, 30, 30])

        bn.backward([0])
        assert_allclose(bn.errorProp.a, [0, 0, 0])

        bn.backward([-10])
        assert_allclose(bn.errorProp.a, [-10, -10, -10])

        bn.backward([3])
        assert_allclose(bn.errorProp.a, [3, 3, 3])


    def test_backwardFork(self):
        g = gt.Graph()
        g.add_vertex(3)
        g.add_edge(g.vertex(0), g.vertex(1))
        g.add_edge(g.vertex(0), g.vertex(2))
        n = Net(1, 2, g)

        bn = BackpropagationNet(n)

        # Init activation functions
        for v in n.g.vertices():
            n.activation[v] = Identity()

        # All weights  == 0
        assert_allclose(bn.errorProp.a, [0, 0, 0])

        bn.backward([1, 1])
        assert_allclose(bn.errorProp.a, [0, 1, 1])

        bn.backward([-100, -100])
        assert_allclose(bn.errorProp.a, [0, -100, -100])

        # Init weights
        for e in n.g.edges():
            n.weightProp[e] = 1.0

        # Tests
        bn.backward([1, 1])
        assert_allclose(bn.errorProp.a, [2, 1, 1])

        bn.backward([30, 30])
        assert_allclose(bn.errorProp.a, [60, 30, 30])

        bn.backward([0, 0])
        assert_allclose(bn.errorProp.a, [0, 0, 0])

        bn.backward([-10, -8])
        assert_allclose(bn.errorProp.a, [-18, -10, -8])

        bn.backward([3, 57])
        assert_allclose(bn.errorProp.a, [60, 3, 57])
