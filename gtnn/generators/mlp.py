import numpy as np

from gtnn.network.activation import LogSigmoid
from gtnn.network.net import Net


def mlp(sizes,
        weightGenerator=np.random.random,
        biasGenerator=np.random.random,
        activationFunction=LogSigmoid(-1, 1)):
    r"""Generate Multilayer Perceptron (MLP) and return as a Net

    :param sizes: Size of each layer
    :type sizes: list of integers
    :param functor weightGenerator: Functor for generating weights
    :param functor biasGenerator: Functor for generating bias values
    :param activationFunction: Activation function - the same for all neurons
    :type activationFunction: Activation function
    :returns: Net object with the requested architecture
    :rtype: :class:`~gtnn.network.Net`

    """
    n = Net(sizes[0], sizes[-1])
    layerId = n.addVertexProperty("layerId", "short")
    lastLayer = []
    presentLayer = []

    # Create all layers
    for layerIdx, size in enumerate(sizes):
        layerProp = n.addLayer()  # TODO: test layer
        for i in range(size):
            v = n.g.add_vertex()
            layerProp[v] = True
            layerId[v] = layerIdx
            n.activation[v] = activationFunction
            n.biasProp[v] = biasGenerator()

            presentLayer.append(v)
            for l in lastLayer:
                e = n.g.add_edge(l, v)
                n.weightProp[e] = weightGenerator()

        lastLayer = list(presentLayer)
        presentLayer = list()

    # Create a subgraph
    n.prepare()
    return n
