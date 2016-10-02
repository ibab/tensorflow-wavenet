"""Unit tests for the loss function."""
from __future__ import print_function
import json
import numpy as np
import tensorflow as tf
from wavenet import MakeLoss


def MakeLogits():
    # 3 channels (quantization levels)
    # 5 time steps in duration
    # 1 batch item
    array = np.array([ [ [  5.0, -10.0, -10.0],
                         [ 10.0, -10.0, -10.0],
                         [  0.0,  10.0,   0.0],
                         [-10.0,  10.0, -10.0],
                         [-10.0,  10.0, -20.0] ] ])

    return tf.convert_to_tensor(array, tf.float32)


def MakeLabels():
    # 3 channels (quantization levels)
    # 5 time steps in duration
    # 1 batch item
    array = np.array([ [ [ 0.0, 1.0, 0.0],
                         [ 0.0, 0.0, 1.0],
                         [ 0.0, 0.0, 1.0],
                         [ 0.0, 0.0, 1.0],
                         [ 0.0, 0.0, 1.0] ] ])
    return tf.convert_to_tensor(array, tf.float32)


class TestLoss(tf.test.TestCase):
    def setUp(self):
        pass


    def testZeroReceptiveFieldSize(self):
        logits = MakeLogits()
        labels = MakeLabels()

        loss = MakeLoss(labels, logits, quantization_channels = 3,
                                     receptive_field_size = 0)
        with self.test_session() as sess:
            loss_array = sess.run([loss])
        expected =np.array([[ 15.00000095, 20.0, 10.0, 20.0, 30.0]])
        EPSILON = 0.001
        np.testing.assert_allclose(expected, loss_array, rtol=EPSILON)


    def testNonZeroReceptiveFieldSize(self):
        logits = MakeLogits()
        labels = MakeLabels()

        loss = MakeLoss(labels, logits, quantization_channels = 3,
                                     receptive_field_size = 2)
        with self.test_session() as sess:
            loss_array = sess.run([loss])

        # Since we truncate the first receptive_field time elements from the
        # the tensors used to compute the loss, the losses should exclude the
        # first two (since receptive_field_size is set to 2).
        expected =np.array([[ 10.0, 20.0, 30.0]])

        EPSILON = 0.001
        np.testing.assert_allclose(expected, loss_array, rtol=EPSILON)


if __name__ == '__main__':
    tf.test.main()

