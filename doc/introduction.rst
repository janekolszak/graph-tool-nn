Introduction
============
Installation
------------
First the dependencies (graph-tool, numpy) and then:

.. code-block:: bash

    pip install graph-tool-nn

Example
-------
XOR function implementation - multilayer perceptron trained with momentum:

.. code-block:: python

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
