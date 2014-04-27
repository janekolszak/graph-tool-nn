import numpy as np


def __learn(net, learningRate):
    for e in net.g.edges():
        net.weightProp[e] += net.valueProp[e.target()] * \
            learningRate * net.errorProp[e.target()]

    for v in net.g.vertices():
        net.biasProp[v] += learningRate * net.errorProp[v]


def train(net, inputs, outputs, numEpochs=100, learningRate=0.1):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    epochIdx = 0
    while True:
        for inp, out in zip(inputs, outputs):
            netOut = net.forward(inp)
            err = netOut - out
            net.backward(err)
            __learn(net, learningRate)
        epochIdx += 1
        if epochIdx > numEpochs:
            break
