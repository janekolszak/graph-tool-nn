import unittest
import graph_tool.all as gt

from gtnn.generators.multilayer_perceptron import *
from gtnn.learn.backpropagation import BackpropagationNet


class TestMultilayerPerceptron(unittest.TestCase):

    def test_construct(self):
        n = multilayer_perceptron([1,2,2,1])
        n.forward([1])

    def test_forwardChain(self):        
        n = multilayer_perceptron([1,1,1])
        n = BackpropagationNet(n)
        for i in range(100):
            n.forward([10]) 
            n.backward([10])
            print(n)
