import graph_tool.all as gt
import numpy as np

from gtnn.network.activation import LogSigmoid, Identity


class Net(object):
    __valueType = "long double"

    def __init__(self,  nInput, nOutput, graph=gt.Graph()):
        self.g = gt.Graph(graph)
        self.nInput = nInput
        self.nOutput = nOutput
        self.properties = dict()
        self.biasProp = self.addVertexProperty("bias", Net.__valueType)
        self.valueProp = self.addVertexProperty("value", Net.__valueType)
        self.sumProp = self.addVertexProperty("sum", Net.__valueType)
        self.activation = self.addVertexProperty(
            "activation", "python::object")
        self.errorProp = self.addVertexProperty("errorProp", Net.__valueType)

        self.weightProp = self.addEdgeProperty("weight", Net.__valueType)

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
        for vIdx in self.order[:self.nInput]:
            self.activation[self.g.vertex(vIdx)] = Identity()

    def forward(self, inputVals=[]):
        g = self.g
        vp = self.valueProp
        bp = self.biasProp
        wp = self.weightProp
        sm = self.sumProp
        activation = self.activation

        for inpVal, vIdx in zip(inputVals, self.order[:self.nInput]):
            vp[g.vertex(vIdx)] = inpVal

        for vIdx in self.order[self.nInput:]:
            v = g.vertex(vIdx)
            inputs = np.array([vp[e.source()] for e in v.in_edges()])
            weights = np.array([wp[e] for e in v.in_edges()])

            sum = np.sum(inputs * weights) + bp[v]
            sm[v] = sum
            vp[v] = activation[v].value(sum)

        return np.array(
            [vp[g.vertex(vIdx)] for vIdx in self.order[-self.nOutput:]]
        )

    def backward(self, outputErr=[]):
        g = self.g
        ep = self.errorProp
        wp = self.weightProp
        sm = self.sumProp

        activation = self.activation

        for outErr, vIdx in zip(outputErr,
                                self.order[-self.nOutput:]):
            v = g.vertex(vIdx)
            ep[v] = outErr * activation[v].derivative(sm[v])

        # TODO: Change the borders
        for vIdx in reversed(self.order[:-self.nOutput]):
            v = g.vertex(vIdx)
            errors = np.array([ep[e.target()] for e in v.out_edges()])
            weights = np.array([wp[e] for e in v.out_edges()])
            ep[v] = np.sum(errors * weights) * activation[v].derivative(sm[v])

    def __str__(self):
        # TODO: Change output format
        ret = "Net: |V|=" + str(self.g.num_vertices())
        ret += " |E|=" + str(self.g.num_edges()) + "\n"

        for key, prop in self.properties.items():
            ret += key + ": " + str(prop.a) + "\n"
        return ret
