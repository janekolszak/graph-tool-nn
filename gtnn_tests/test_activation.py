import unittest

from gtnn.network.activation import LogSigmoid


class TestActivation(unittest.TestCase):

    def test_logsigmoid(self):
        l = LogSigmoid(-1, 1)
        self.assertLess(l.value(-10), 0)
        self.assertGreater(l.value(10), 0)
        self.assertEqual(l.value(0), 0)
        self.assertLess(l.derivative(-10), 0.5)
        self.assertLess(l.derivative(10), 0.5)
