import numpy as np

# TODO: Rename online


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

            net.weightProp.a += prevWeightChangeProp.a

            # Bias learning
            prevBiasChangeProp.a = -learningRate * net.errorProp.a + \
                momentum * prevBiasChangeProp.a

            net.biasProp.a += prevBiasChangeProp.a

        # End condition
        epochIdx += 1
        if epochIdx > numEpochs:
            break


def miniBatchTrain(net,
                   inputs,
                   outputs,
                   numEpochs=100,
                   learningRate=0.1,
                   momentum=0.6,
                   batchSize=10):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    prevWeightChangeProp = net.addEdgeProperty("previous weight change",
                                               "long double")

    prevBiasChangeProp = net.addVertexProperty("previous bias change",
                                               "long double")

    sumWeightChangeProp = net.addEdgeProperty("sum weight change",
                                              "long double")

    sumBiasChangeProp = net.addVertexProperty("sum bias change",
                                              "long double")

    epochIdx = 0
    while True:
        for idx, (inp, out) in enumerate(zip(inputs, outputs)):
            netOut = net.forward(inp)
            err = netOut - out
            net.backward(err)

            # Summing up the changes in weights and bias
            for e in net.g.edges():
                sumWeightChangeProp[e] += net.valueProp[e.source()] * \
                    net.errorProp[e.target()]

            for v in net.g.vertices():
                sumBiasChangeProp[v] += net.errorProp[v]

            if idx % batchSize == 0:

                # Weights learning:
                for e in net.g.edges():
                    prevWeightChangeProp[e] = -learningRate * \
                        sumWeightChangeProp[e] / batchSize + \
                        prevWeightChangeProp[e] * momentum

                net.weightProp.a += prevWeightChangeProp.a

                # Bias learning
                prevBiasChangeProp.a = -learningRate * \
                    sumBiasChangeProp.a / batchSize + \
                    momentum * prevBiasChangeProp.a

                net.biasProp.a += prevBiasChangeProp.a

                # Reset the changes
                sumWeightChangeProp.a = np.zeros(net.g.num_edges())
                sumBiasChangeProp.a = np.zeros(net.g.num_vertices())

        # End condition
        epochIdx += 1
        if epochIdx > numEpochs:
            break


def batchTrain(net,
               inputs,
               outputs,
               numEpochs=100,
               learningRate=0.1,
               momentum=0.6):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    batchSize = len(inputs)
    prevWeightChangeProp = net.addEdgeProperty("previous weight change",
                                               "long double")

    prevBiasChangeProp = net.addVertexProperty("previous bias change",
                                               "long double")

    sumWeightChangeProp = net.addEdgeProperty("sum weight change",
                                              "long double")

    sumBiasChangeProp = net.addVertexProperty("sum bias change",
                                              "long double")

    epochIdx = 0
    while True:
        for idx, (inp, out) in enumerate(zip(inputs, outputs)):
            netOut = net.forward(inp)
            err = netOut - out
            net.backward(err)

            # Summing up the changes in weights and bias
            for e in net.g.edges():
                sumWeightChangeProp[e] += net.valueProp[e.source()] * \
                    net.errorProp[e.target()]

            for v in net.g.vertices():
                sumBiasChangeProp[v] += net.errorProp[v]

        # Weights learning:
        for e in net.g.edges():
            prevWeightChangeProp[e] = -learningRate * \
                sumWeightChangeProp[e] / batchSize + \
                prevWeightChangeProp[e] * momentum

        net.weightProp.a += prevWeightChangeProp.a

        # Bias learning
        prevBiasChangeProp.a = -learningRate * \
            sumBiasChangeProp.a / batchSize + \
            momentum * prevBiasChangeProp.a

        net.biasProp.a += prevBiasChangeProp.a

        # Reset the changes
        sumWeightChangeProp.a = np.zeros(net.g.num_edges())
        sumBiasChangeProp.a = np.zeros(net.g.num_vertices())

        # End condition
        epochIdx += 1
        if epochIdx > numEpochs:
            break
