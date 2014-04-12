import unittest
import graph_tool.all as gt

from gtnn.network.net import Net


class TestNet(unittest.TestCase):

    def test_net(self):
        g = gt.Graph()
        g.add_vertex(4)
        g.add_edge(g.vertex(0), g.vertex(1))
        g.add_edge(g.vertex(0), g.vertex(2))
        g.add_edge(g.vertex(1), g.vertex(3))
        g.add_edge(g.vertex(2), g.vertex(3))
        g.add_vertex(1)
        g.add_edge(g.vertex(4), g.vertex(0))
        n = Net(1, 1,g)
        n.prepare()
        n.forward([34])
