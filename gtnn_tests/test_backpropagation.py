import unittest
import graph_tool.all as gt

from gtnn.learn.backpropagation import BackpropagationNet
from gtnn.network.net import Net

class TestBackpropagation(unittest.TestCase):

    def test_backpropagation(self):
        g = gt.Graph()
        g.add_vertex(4)
        g.add_edge(g.vertex(0), g.vertex(1))
        g.add_edge(g.vertex(0), g.vertex(2))
        g.add_edge(g.vertex(1), g.vertex(3))
        g.add_edge(g.vertex(2), g.vertex(3))
        g.add_vertex(1)
        g.add_edge(g.vertex(4), g.vertex(0))
        n = Net(1, 1,g)
        for e in g.edges():
            n.weightProp[e] = 1
        n.prepare()

        bn = BackpropagationNet(n)

        bn.backward()
        # print(n.order)
        # n.forward([34])
        # forward(n)

