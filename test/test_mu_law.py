import numpy as np
import tensorflow as tf

from wavenet_ops import mu_law_encode, mu_law_decode


class TestMuLaw(tf.test.TestCase):

    def testEncodeDecode(self):
        x = np.linspace(-1, 1, 1000).astype(np.float32)
        channels = 256

        # Test whether decoded signal is roughly equal to
        # what was encoded before
        with self.test_session() as sess:
            encoded = mu_law_encode(x, channels)
            x1 = sess.run(mu_law_decode(encoded, channels))

        self.assertAllClose(x, x1, rtol=1e-1, atol=0.05)

        # Make sure that re-encoding leaves the waveform invariant
        with self.test_session() as sess:
            encoded = mu_law_encode(x1, channels)
            x2 = sess.run(mu_law_decode(encoded, channels))

        self.assertAllClose(x1, x2)

    def testEncodeIsSurjective(self):
        x = np.linspace(-1, 1, 10000).astype(np.float32)
        channels = 123
        with self.test_session() as sess:
            encoded = sess.run(mu_law_encode(x, channels))
        self.assertEqual(len(np.unique(encoded)), channels)


if __name__ == '__main__':
    tf.test.main()
