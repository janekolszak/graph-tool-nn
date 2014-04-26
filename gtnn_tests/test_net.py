import unittest
from numpy.testing import assert_allclose
import graph_tool.all as gt

from gtnn.network.net import Net
from gtnn.network.activation import Identity


class TestNet(unittest.TestCase):

    def test_construct(self):
        g = gt.Graph()
        g.add_vertex(4)
        g.add_edge(g.vertex(0), g.vertex(1))
        g.add_edge(g.vertex(0), g.vertex(2))
        g.add_edge(g.vertex(1), g.vertex(3))
        g.add_edge(g.vertex(2), g.vertex(3))
        g.add_vertex(1)
        g.add_edge(g.vertex(4), g.vertex(0))
        n = Net(1, 1, g)
        return n

    def test_forwardChain(self):
        g = gt.Graph()
        g.add_vertex(3)
        g.add_edge(g.vertex(0), g.vertex(1))
        g.add_edge(g.vertex(1), g.vertex(2))
        n = Net(1, 1, g)

        # Init activation functions
        for v in n.g.vertices():
            n.activation[v] = Identity()

        # All weights and bias == 0
        assert_allclose(n.forward([1.0]), [0.0])

        # Init weights
        for e in n.g.edges():
            n.weightProp[e] = 1.0
        assert_allclose(n.forward([0.0]), [0.0])
        assert_allclose(n.forward([1.0]), [1.0])
        assert_allclose(n.forward([12.0]), [12.0])
        assert_allclose(n.forward([-12.0]), [-12.0])

        # Init bias
        for v in n.g.vertices():
            n.biasProp[v] = 1.0
        assert_allclose(n.forward([0.0]), [2.0])
        assert_allclose(n.forward([1.0]), [3.0])

    def test_forwardFork(self):
        g = gt.Graph()
        g.add_vertex(3)
        g.add_edge(g.vertex(0), g.vertex(1))
        g.add_edge(g.vertex(0), g.vertex(2))
        n = Net(1, 2, g)

        # Init activation functions
        for v in n.g.vertices():
            n.activation[v] = Identity()

        # All weights and bias == 0
        assert_allclose(n.forward([1.0]), [0.0, 0.0])

        # Init weights
        for e in n.g.edges():
            n.weightProp[e] = 1.0
        assert_allclose(n.forward([0.0]), [0.0, 0.0])
        assert_allclose(n.forward([1.0]), [1.0, 1.0])
        assert_allclose(n.forward([12.0]), [12.0, 12.0])
        assert_allclose(n.forward([-12.0]), [-12.0, -12.0])

        # Init bias
        for v in n.g.vertices():
            n.biasProp[v] = 1.0
        assert_allclose(n.forward([0.0]), [1.0, 1.0])
        assert_allclose(n.forward([1.0]), [2.0, 2.0])
