import graph_tool.all as gt
import numpy as np

from gtnn.network.activation import LogSigmoid


class Net(object):
    __valueType = "long double"

    def __init__(self,  nInput=1, nOutput=1, graph=gt.Graph()):
        self.g = gt.Graph(graph)
        self.nInput = nInput
        self.nOutput = nOutput
        self.properties = dict()
        self.biasProp = self.addVertexProperty("bias", Net.__valueType)
        self.valueProp = self.addVertexProperty("value", Net.__valueType)
        self.sumProp = self.addVertexProperty("sum", Net.__valueType)
        self.activation = self.addVertexProperty("activation", "python::object")
        self.weightProp = self.addEdgeProperty("weight", Net.__valueType)
        for v in self.g.vertices():
            self.activation[v] = LogSigmoid(-1, 1)
        self.prepare()

    def addVertexProperty(self, name, typeName):
        self.properties[name] = self.g.new_vertex_property(typeName)
        return self.properties[name]

    def addEdgeProperty(self, name, typeName):
        self.properties[name] = self.g.new_edge_property(typeName)
        return self.properties[name]

    def prepare(self):
        """
        Computes ans saves the topological sort of the graph for future use.
        """
        self.order = np.array(gt.topological_sort(self.g)[::-1])

    def forward(self, inputVals=[]):
        g = self.g
        vp = self.valueProp
        bp = self.biasProp
        wp = self.weightProp
        sm = self.sumProp
        activation = self.activation

        for inpVal, vIdx in zip(inputVals, self.order):
            vp[g.vertex(vIdx)] = inpVal

        for vIdx in self.order[self.nInput:]:
            v = g.vertex(vIdx)
            inputs = np.array([vp[e.source()] for e in v.in_edges()])
            weights = np.array([wp[e] for e in v.in_edges()])

            sum = np.sum(inputs * weights) + bp[v]
            sm[v] = sum
            vp[v] = activation[v].value(sum)

        return np.array(vp.a[-self.nOutput:])

    def __str__(self):
        ret = "Net: |V|=" + str(self.g.num_vertices())
        ret += " |E|=" + str(self.g.num_edges()) + "\n"

        for key, prop in self.properties.items():
            ret += key + ": " + str(prop.a) + "\n"
        return ret
