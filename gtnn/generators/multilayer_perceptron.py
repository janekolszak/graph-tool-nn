import numpy as np
import graph_tool.all as gt

from gtnn.network.activation import LogSigmoid
from gtnn.network.net import Net


def multilayer_perceptron(sizes=[1, 1],
                          weightGenerator=np.random.random,
                          biasGenerator=np.random.random,
                          activationFunction=LogSigmoid(-1, -1)):

    n = Net(sizes[0], sizes[-1])
    layerId = n.addVertexProperty("layerId", "short")
    lastLayer = []
    presentLayer = []

    for layerIdx, size in enumerate(sizes):
        for i in range(size):
            v = n.g.add_vertex()
            layerId[v] = layerIdx
            n.activation[v] = activationFunction
            n.valueProp[v] = 0.0
            n.biasProp[v] = biasGenerator()

            presentLayer.append(v)
            for l in lastLayer:
                e = n.g.add_edge(l, v)
                n.weightProp[e] = weightGenerator()

        lastLayer = list(presentLayer)
        presentLayer = list()
        
    n.prepare()
    return n