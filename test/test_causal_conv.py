"""Unit tests for the causal_conv op."""

import numpy as np
import tensorflow as tf

from wavenet import time_to_batch, batch_to_time, causal_conv


class TestCausalConv(tf.test.TestCase):

    def testCausalConv(self):
        """Tests that the op is equivalent to a numpy implementation."""
        x1 = np.arange(1, 21, dtype=np.float32)
        x = np.append(x1, x1)
        x = np.reshape(x, [2, 20, 1])
        f = np.reshape(np.array([1, 1], dtype=np.float32), [2, 1, 1])
        out = causal_conv(x, f, 4)

        with self.test_session() as sess:
            result = sess.run(out)

        # Causal convolution using numpy
        ref = np.convolve(x1, [1, 0, 0, 0, 1], mode='valid')
        ref = np.append(ref, ref)
        ref = np.reshape(ref, [2, 16, 1])

        self.assertAllEqual(result, ref)

    def testNoTimeShift(self):
        """Tests that the convolution does not introduce a time shift.

        We give it a time series, choose a filter that should be the identity,
        and assert that the output is not shifted at all relative to the input.
        """
        # Input to filter is a time series of values 1..10
        x = np.arange(1, 11, dtype=np.float32)
        # Reshape the input: shape is batch item x duration x channels = 1x10x1
        x = np.reshape(x, [1, 10, 1])
        # Default shape ordering for conv filter = HWIO for 2d. Since we use
        # 1d, this just becomes WxIxO where:
        #   W = width AKA number of time steps in time series = 2
        #   I = input channels = 1
        #   O = output channels = 1
        # Since the filter is size 2, for it to be identity-preserving, one
        # value is 1.0, the other 0.0
        filter = np.reshape(np.array([0.0, 1.0], dtype=np.float32), [2, 1, 1])

        x_padded = np.pad(x, [[0, 0], [2, 0], [0, 0]], 'constant')

        # Compute the output
        out = causal_conv(x_padded, filter, dilation=2)

        with self.test_session() as sess:
            result = sess.run(out)

        # The shapes should be the same.
        self.assertAllEqual(result.shape, x.shape)

        # The output time series should be identical to the input series.
        self.assertAllEqual(result, x)


if __name__ == '__main__':
    tf.test.main()
