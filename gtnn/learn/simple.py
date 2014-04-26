import numpy as np

from gtnn.learn.backpropagation import BackpropagationNet


def __learn(net, learningRate):
    for e in net.net.g.edges():
        net.net.weightProp[e] += net.net.valueProp[e.target()] * \
            learningRate * net.errorProp[e.target()]

    for v in net.net.g.vertices():
        net.net.biasProp[v] += learningRate * net.errorProp[v]


def train(net, inputs, outputs, numEpochs=100, learningRate=0.1):
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    n = BackpropagationNet(net)

    epochIdx = 0
    while True:
        for inp, out in zip(inputs, outputs):
            netOut = n.forward(inp)
            err = netOut - out
            n.backward(err)
            __learn(n, learningRate)
        epochIdx += 1
        if epochIdx > numEpochs:
            break
