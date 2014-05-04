import numpy as np


def train(net, inputs, outputs, numEpochs=100, learningRate=0.1, momentum=0.6):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    prevWeightChangeProp = net.addEdgeProperty("previous weight change",
                                               "long double")

    prevBiasChangeProp = net.addVertexProperty("previous bias change",
                                               "long double")
    epochIdx = 0
    while True:
        for inp, out in zip(inputs, outputs):
            netOut = net.forward(inp)
            err = netOut - out
            net.backward(err)

            # Weights learning:
            for e in net.g.edges():
                prevWeightChangeProp[e] = -net.valueProp[e.source()] * \
                    learningRate * net.errorProp[e.target()] + \
                    prevWeightChangeProp[e] * momentum

                net.weightProp[e] += prevWeightChangeProp[e]

            # Bias learning
            for v in net.g.vertices():
                prevBiasChangeProp[v] = -learningRate * net.errorProp[v] + \
                    momentum * prevBiasChangeProp[v]

                net.biasProp[v] += prevBiasChangeProp[v]

        # End condition
        epochIdx += 1
        if epochIdx > numEpochs:
            break
