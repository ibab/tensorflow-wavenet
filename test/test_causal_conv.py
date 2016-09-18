
import tensorflow as tf
import numpy as np
from wavenet_ops import time_to_batch, batch_to_time, causal_conv

class TestCausalConv(tf.test.TestCase):

    def testCausalConv(self):
        x1 = np.arange(1, 21, dtype=np.float32)
        x = np.append(x1, x1)
        x = np.reshape(x, [2, 1, 20, 1])
        f = np.reshape(np.array([1, 1], dtype=np.float32), [1, 2, 1, 1])
        out = causal_conv(x, f, 4)

        with self.test_session() as sess:
            result = sess.run(out)

        # Causal convolution using numpy
        ref = np.convolve(x1, [1, 0, 0, 0, 1])[:-4]
        ref = np.append(ref, ref)
        ref = np.reshape(ref, [2, 1, 20, 1])

        self.assertAllEqual(result, ref)

