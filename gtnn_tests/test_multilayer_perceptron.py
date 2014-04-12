import unittest
import graph_tool.all as gt

from gtnn.generators.multilayer_perceptron import *


class TestNet(unittest.TestCase):

    def test_mpl(self):
        n = multilayer_perceptron([1,2,2,1])

