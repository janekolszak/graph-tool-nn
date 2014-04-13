import graph_tool.all as gt
import numpy as np

from gtnn.network.net import Net


class BackpropagationNet(object):

    def __init__(self, net):
        self.net = net
        self.errorProp = net.addVertexProperty("errorProp", "long double")

    def prepare(self):
        self.net.prepare()

    def forward(self, inptut):
        self.net.forward(input)

    def backward(self, outputErr=[]):
        g = self.net.g
        ep = self.errorProp
        wp = self.net.weightProp
        sm = self.net.sumProp
        activation = self.net.activation

        for outErr, vIdx in zip(outputErr,
                                self.net.order[-self.net.nOutput:]):
            ep[g.vertex(vIdx)] = outErr

        for vIdx in reversed(self.net.order):
            v = g.vertex(vIdx)

            errors = np.array([ep[e.target()] for e in v.out_edges()])
            weights = np.array([wp[e] for e in v.out_edges()])
            ep[v] = np.sum(errors * weights)*activation[v].derivative(sm[v])
