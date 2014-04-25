import unittest
import graph_tool.all as gt

from gtnn.generators.multilayer_perceptron import multilayer_perceptron
from gtnn.learn.simple import train


class TestSimple(unittest.TestCase):

    def test_simple(self):
        inp = [[1]]
        out = [[1]]
        n = multilayer_perceptron([1, 1, 1])

        train(net=n, inputs=inp, outputs=out,
              numEpochs=1000, learningRate=0.08)

        for i, o in zip(inp, out):
            print(str(n.forward(i)) + " " + str(o))
