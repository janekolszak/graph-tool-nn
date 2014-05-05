import numpy as np

from gtnn.network.activation import LogSigmoid
from gtnn.network.net import Net


def mlp(sizes,
        weightGenerator=np.random.random,
        biasGenerator=np.random.random,
        activationFunction=LogSigmoid(-1, 1)):
    r"""Geneerate Multilayer Perceptron (MLP) and return as a Net

    Parameters
    ----------
    sizes : list of integers
        Size of each layer
    weightGenerator : functor
        Functor for generating weights
    biasGenerator : functor
        Functor for generating bias values
    activationFunction : activation function
        Activation function - the same for all neurons

    Returns
    -------
    net : :class:`~gtnn.network.Net`
        Net object with the requested architecture

    Notes
    -----
    Creates MLP

    """
    n = Net(sizes[0], sizes[-1])
    layerId = n.addVertexProperty("layerId", "short")
    lastLayer = []
    presentLayer = []

    # Create all layers
    for layerIdx, size in enumerate(sizes):
        for i in range(size):
            v = n.g.add_vertex()
            layerId[v] = layerIdx
            n.activation[v] = activationFunction
            n.biasProp[v] = biasGenerator()

            presentLayer.append(v)
            for l in lastLayer:
                e = n.g.add_edge(l, v)
                n.weightProp[e] = weightGenerator()

        lastLayer = list(presentLayer)
        presentLayer = list()

    n.prepare()
    return n
