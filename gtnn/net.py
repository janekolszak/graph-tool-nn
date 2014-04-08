import graph_tool.all as gt
import numpy as np


class Net(object):

    def __init__(self, graph, nInput, nOutput, valueType="long double"):
        self.g = gt.Graph(graph)
        self.__nInput = nInput
        self.__nOutput = nOutput
        self.biasProp = self.g.new_vertex_property(valueType)
        self.valueProp = self.g.new_vertex_property(valueType)
        self.weightProp = self.g.new_edge_property(valueType)
        self.prepare()

    def prepare(self):
        """
        Computes ans saves the topological sort of the graph for future use.
        """
        self.order = gt.topological_sort(self.g)[::-1]

    def forward(self, inputVals=[]):
        g = self.g
        vp = self.valueProp
        bp = self.biasProp
        wp = self.weightProp

        for inpVal, vIdx in zip(inputVals, self.order):
            vp[g.vertex(vIdx)] = inpVal

        for vIdx in self.order:
            v = g.vertex(vIdx)

            inputs = np.array([vp[e.source()] for e in v.in_edges()])
            weights = np.array([wp[e] for e in v.in_edges()])

            vp[v] = np.sum(inputs * weights) + bp[v]
            print(vp[v])

        return np.array(vp.a[-self.__nOutput:])

def forward(net):
    return net.forward()

def main():
    g = gt.Graph()
    g.add_vertex(4)
    g.add_edge(g.vertex(0), g.vertex(1))
    g.add_edge(g.vertex(0), g.vertex(2))
    g.add_edge(g.vertex(1), g.vertex(3))
    g.add_edge(g.vertex(2), g.vertex(3))
    g.add_vertex(1)
    g.add_edge(g.vertex(4), g.vertex(0))
    n = Net(g, 1, 1)
    n.prepare()
    # gt.graph_draw(g)
    # print(n.order)
    n.forward([34])
    forward(n)

if __name__ == "__main__":
    main()
