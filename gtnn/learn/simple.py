import numpy as np

from gtnn.learn.backpropagation import BackpropagationNet


def __learn(net, learningRate):
    for e in net.net.g.edges():
        # print("LER " + str(net.net.valueProp[e.target()]),
        #       str(learningRate),
        #       str(net.errorProp[e.target()]))

        net.net.weightProp[e] += net.net.valueProp[e.target()] * \
            learningRate * net.errorProp[e.target()]

    for v in net.net.g.vertices():
        net.net.biasProp[v] += learningRate * net.errorProp[v]


def train(net, inputs, outputs, numEpochs=1, learningRate=0.1):
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    n = BackpropagationNet(net)

    epochIdx = 0
    while True:
        # print(n.net)
        for inp, out in zip(inputs, outputs):
            # print(str(inp))
            # print(str(out))
            netOut = n.forward(inp)
            # print("TR " + str(netOut))
            err = out - netOut
            print("ERR " + str(err))
            n.backward(err)
            __learn(n, learningRate)

        epochIdx += 1
        if epochIdx > numEpochs:
            break
