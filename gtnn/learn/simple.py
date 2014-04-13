from gtnn.learn.backpropagation import BackPropagationNet


def __learn(net, learningRate):
    for e in net.g.edges():
        net.weightProp[e] += net.valueProp * \
            learningRate * net.errorProp[e.target()]


def train(net, inputs, outputs, numEpochs=1, learningRate=0.8):
    n = BackPropagationNet(net)
    epochIdx = 0
    while True:
        for inp, out in zip(inputs, outputs):
            netOut = n.forward(inp)
            err = out - netOut
            n.backward(err)
            __learn(n)

        epochIdx += 1
        if epochIdx > numEpochs:
            break
