import numpy as np
import tensorflow as tf

from wavenet_ops import encode, decode

class TestMuLaw(tf.test.TestCase):

    def testEncodeDecode(self):

        x = np.linspace(-1, 1, 1000).astype(np.float32)
        channels = 256

        # Test whether decoded signal is roughly equal to
        # what was encoded before
        with self.test_session() as sess:
            x1 = sess.run(decode(encode(x, channels), channels))

        self.assertAllClose(x, x1, rtol=1e-1, atol=0.05)

        # Make sure that re-encoding leaves the waveform invariant
        with self.test_session() as sess:
            x2 = sess.run(decode(encode(x1, channels), channels))

        self.assertAllClose(x1, x2)

if __name__ == '__main__':
    tf.test.main()
