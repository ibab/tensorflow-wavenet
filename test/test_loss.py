"""Unit tests for the loss function."""
from __future__ import print_function
import json
import numpy as np
import tensorflow as tf
from wavenet import MakeLoss, ComputeReceptiveFieldSize


def MakeLogits():
    # 1 batch item
    # 5 time steps in duration
    # 3 channels (quantization levels)
    array = np.array([[[5.0,   -10.0, -10.0],
                       [10.0,  -10.0, -10.0],
                       [0.0,    10.0,   0.0],
                       [-10.0,  10.0, -10.0],
                       [-10.0,  10.0, -20.0]]])

    return tf.convert_to_tensor(array, tf.float32)


def MakeLabels():
    # 1 batch item
    # 5 time steps in duration
    # 3 channels (quantization levels)
    array = np.array([[[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 0.0, 1.0]]])
    return tf.convert_to_tensor(array, tf.float32)


class TestLoss(tf.test.TestCase):
    def setUp(self):
        pass

    def testZeroReceptiveFieldSize(self):
        logits = MakeLogits()
        labels = MakeLabels()

        loss = MakeLoss(labels, logits, quantization_channels=3,
                        receptive_field_size=0)
        with self.test_session() as sess:
            loss_array = sess.run([loss])
        expected = np.array([[15.0, 20.0, 10.0, 20.0, 30.0]])
        EPSILON = 0.001
        np.testing.assert_allclose(expected, loss_array, rtol=EPSILON)

    def testNonZeroReceptiveFieldSize(self):
        logits = MakeLogits()
        labels = MakeLabels()

        loss = MakeLoss(labels, logits, quantization_channels=3,
                        receptive_field_size=2)
        with self.test_session() as sess:
            loss_array = sess.run([loss])

        # Since we truncate the first receptive_field time elements from the
        # the tensors used to compute the loss, the losses should exclude the
        # first two (since receptive_field_size is set to 2) that we would
        # otherwise have.
        expected = np.array([[10.0, 20.0, 30.0]])

        EPSILON = 0.001
        np.testing.assert_allclose(expected, loss_array, rtol=EPSILON)

    def testComputeReceptiveFieldSize_4(self):
        dilations = [1, 2, 4]
        computed_size = ComputeReceptiveFieldSize(dilations)
        self.assertEqual(computed_size, 8)

    def testComputeReceptiveFieldSize_256_256_256(self):
        dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256,
                     1, 2, 4, 8, 16, 32, 64, 128, 256,
                     1, 2, 4, 8, 16, 32, 64, 128, 256]
        computed_size = ComputeReceptiveFieldSize(dilations)
        self.assertEqual(computed_size, 512 + 511 + 511)


if __name__ == '__main__':
    tf.test.main()
