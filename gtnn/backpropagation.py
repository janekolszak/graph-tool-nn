import graph_tool.all as gt
import numpy as np

from net import Net


class BackpropagationNet(Net):

    def __init__(self, graph, nInput, nOutput):
        super().__init__(graph, nInput, nOutput)
        self.__nInput = nInput
        self.__nOutput = nOutput
        self.errorProp = self.g.new_vertex_property("long double")

    def prepare(self):
        super().prepare()

    def backward(self, outputErr=[]):
        g = self.g
        ep = self.errorProp
        wp = self.weightProp

        for outErr, vIdx in zip(outputErr,
                                self.order[-self.__nOutput:]):
            ep[g.vertex(vIdx)] = outErr

        for vIdx in reversed(self.order):
            v = g.vertex(vIdx)

            errors = np.array([ep[e.target()] for e in v.out_edges()])
            weights = np.array([wp[e] for e in v.out_edges()])
            ep[v] = np.sum(errors * weights)


def main():
    g = gt.Graph()
    g.add_vertex(4)
    g.add_edge(g.vertex(0), g.vertex(1))
    g.add_edge(g.vertex(0), g.vertex(2))
    g.add_edge(g.vertex(1), g.vertex(3))
    g.add_edge(g.vertex(2), g.vertex(3))
    g.add_vertex(1)
    g.add_edge(g.vertex(4), g.vertex(0))

    n = BackpropagationNet(g, 1, 1)
    for e in g.edges():
        n.weightProp[e] = 1

    n.prepare()
    n.backward()
    # print(n.order)
    # n.forward([34])
    # forward(n)

if __name__ == "__main__":
    main()
