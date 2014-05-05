import numpy as np


def train(net, inputs, outputs, numEpochs=100, learningRate=0.1):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    epochIdx = 0
    while True:
        for inp, out in zip(inputs, outputs):
            netOut = net.forward(inp)
            err = netOut - out
            net.backward(err)

            # Weights learning:
            for e in net.g.edges():
                net.weightProp[e] -= net.valueProp[e.source()] * \
                    learningRate * net.errorProp[e.target()]

            # Bias learning
            for v in net.g.vertices():
                net.biasProp[v] -= learningRate * net.errorProp[v]

        # End condition
        epochIdx += 1
        if epochIdx > numEpochs:
            break
