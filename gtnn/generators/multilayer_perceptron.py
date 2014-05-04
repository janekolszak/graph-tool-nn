import numpy as np
import graph_tool.all as gt

from gtnn.network.activation import LogSigmoid
from gtnn.network.net import Net


def multilayer_perceptron(sizes,
                          weightGenerator=np.random.random,
                          biasGenerator=np.random.random,
                          activationFunction=LogSigmoid(-1, 1)):

    n = Net(sizes[0], sizes[-1])
    layerId = n.addVertexProperty("layerId", "short")
    lastLayer = []
    presentLayer = []

    # Create all layers
    for layerIdx, size in enumerate(sizes):
        for i in range(size):
            v = n.g.add_vertex()
            layerId[v] = layerIdx
            # n.valueProp[v] = 0.0
            n.activation[v] = activationFunction
            n.biasProp[v] = biasGenerator()

            presentLayer.append(v)
            for l in lastLayer:
                e = n.g.add_edge(l, v)
                n.weightProp[e] = weightGenerator()

        lastLayer = list(presentLayer)
        presentLayer = list()

    # Fill specific stuff for hidden and output layers
    # for v in n.g.vertices():

    n.prepare()
    return n