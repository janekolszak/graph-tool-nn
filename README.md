Graph-tool Neural Networks
==========================

Graph-tool is a great open source tool for creating, using and analyzing graphs. It's a python
library with C++ bindings, uses boost::graph under the hood and seems to be pretty fast
(http://graph-tool.skewed.de/).

Graph-tool Neural Networks (gtnn) is an implementation of ANN on top of graph-tool. It makes
researching neural networks nice&easy. You can create custom nets, train, analyze and plot them.

Installation
============
````bash
pip install graph-tool-nn
````

Usage
=====
````python
from numpy.testing import assert_allclose
import numpy as np

from gtnn.generators.mlp import mlp
from gtnn.learn.momentum import train
from gtnn.network.activation import LogSigmoid

inp = [[0, 0], [1, 0], [0, 1], [1, 1]]
out = [[0], [1], [1], [0]]

n = mlp(sizes=[2, 2, 1],
        weightGenerator=np.random.random,
        biasGenerator=np.random.random,
        activationFunction=LogSigmoid(0, 1))

train(net=n, 
      inputs=inp, 
      outputs=out,
      numEpochs=1000, 
      learningRate=0.3, 
      momentum=0.8)

assert_allclose([n.forward(i) for i in inp], out, atol=0.1)

````
