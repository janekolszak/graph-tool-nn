import graph_tool.all as gt
import numpy as np


class LayeredNet(object):

    """This is a wrapper for the graph_tool.Graph object.
    One can register vertex/edge properties and use them in algorithm implementation.
    It also implements basic forward and backward operations.

    One can create a layered architecture with addLayer() method.
    Layers can overlap (can have some common neurons).
    The first layer is the input layer and the last one is the output layer.
    The output of the n-th layer is computed after the n-1 layer.

    Activation function is computed from the whole layer, so various non-trivial
    applications are possible. For example we create a Softmax layer only by
    introducing a suitable activation function..
    """

    def __init__(self,  nInput, nOutput, graph=gt.Graph()):
        self.g = gt.Graph(graph)
        self.properties = dict()
        self.biasProp = self.addVertexProperty("bias", "long double")
        self.valueProp = self.addVertexProperty("value", "long double")
        self.sumProp = self.addVertexProperty("sum", "long double")
        self.activation = self.addVertexProperty("activation", "python::object")
        self.errorProp = self.addVertexProperty("errorProp", "long double")
        self.weightProp = self.addEdgeProperty("weight", "long double")

        self.prepare()

    def addVertexProperty(self, name, typeName):
        self.properties[name] = self.g.new_vertex_property(typeName)
        return self.properties[name]

    def addEdgeProperty(self, name, typeName):
        self.properties[name] = self.g.new_edge_property(typeName)
        return self.properties[name]

    def addLayer(self):
        self.layers.append(self.g.new_vertex_property("bool"))
        return self.layers[-1]

    def prepare(self):
        """ Run after the layer structure is changed
        Creates helper GrahView objects"""
        self.layerGraphs = []
        for layerFilter in zip(self.layers[:-1], self.layers[1:]):
            self.layerGraphs.append(gt.GraphView(self.g,
                                                 vfilt=layerFilter))

    def forward(self, inputVals=[]):
        vp = self.valueProp
        bp = self.biasProp
        wp = self.weightProp
        sm = self.sumProp
        g = self.g
        activation = self.activation

        # Fill the input neurons
        # First layer is assumed to be the input layer.
        for inpVal, v in zip(inputVals,
                             self.layerGraphs[0].vertices()):
            vp[v] = inpVal

        for layerGraph in self.layerGraphs[1:]:
            # Compute the sum of inputs for every neuron in this layer
            for vLr in layerGraph.vertices():
                v = g.vertex(vLr)
                inputs = np.array([vp[e.source()] for e in v.in_edges()])
                weights = np.array([wp[e] for e in v.in_edges()])

                sm[v] = np.sum(inputs * weights) + bp[v]

            # NOTE: The whole layer has the same activation function.
            #       It could be a graph property
            a = activation[layerGraph.vertices().next()]
            activations = a.value([sm[vLr] for vLr in layerGraph.vertices()])

            # Fill the layer's values
            for v, act in zip(layerGraph.vertices(), activations):
                vp[v] = act

        return np.array([vp[v] for v in self.layerGraphs[-1].vertices()])

    def backward(self, outputErr=[]):
        g = self.g
        ep = self.errorProp
        wp = self.weightProp
        sm = self.sumProp
        activation = self.activation
        outLayer = self.layerGraphs[-1]

        # Compute the derivative on the whole layer:
        # NOTE: The whole layer has the same activation function.
        #       It could be a graph property
        ap = activation[outLayer.vertices().next()]
        derivatives = ap.derivative([sm[vLr] for vLr in outLayer.vertices()])

        # Fill the output neuron's errors
        for v, outErr, dv in zip(outLayer.vertices(),
                                 outputErr,
                                 derivatives):
            ep[v] = outErr * dv

        # Fill the errors in the other layers
        for layerGraph in reversed(self.layerGraphs[:-1]):
            for vLr in layerGraph.vertices():
                v = g.vertex(vLr)
                errors = np.array([ep[e.target()] for e in v.out_edges()])
                weights = np.array([wp[e] for e in v.out_edges()])
                ep[v] = np.sum(errors * weights)

            ap = activation[layerGraph.vertices().next()]
            derivatives = ap.derivative([sm[vLr] for vLr in layerGraph.vertices()])

            # Correct the errors with the derivative
            for vLr, dv in zip(layerGraph.vertices(),
                               derivatives):
                ep[vLr] *= dv

    def __str__(self):
        # TODO: Change output format
        ret = "Net: |V|=" + str(self.g.num_vertices())
        ret += " |E|=" + str(self.g.num_edges()) + "\n"

        for key, prop in self.properties.items():
            ret += key + ": " + str(prop.a) + "\n"
        return ret
