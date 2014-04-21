# import unittest
# import graph_tool.all as gt

# from gtnn.network.net import Net
# from gtnn.generators.multilayer_perceptron import multilayer_perceptron
# from gtnn.learn.simple import train


# class TestSimple(unittest.TestCase):

#     def test_simple(self):
#         inp = [[0, 0],
#                [0, 1],
#                [1, 0],
#                [1, 1]]
#         out = [[0], [1], [1], [0]]
#         n = multilayer_perceptron([2, 2, 1])

#         train(net=n, inputs=inp, outputs=out,
#               numEpochs=200, learningRate=0.08)

#         for i, o in zip(inp, out):
#             print(str(n.forward(i)) + " " + str(o))
