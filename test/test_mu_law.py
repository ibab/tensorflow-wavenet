import tensorflow as tf
import numpy as np
from wavenet import WaveNet

class TestMuLaw(tf.test.TestCase):

    def testEncodeDecode(self):

        net = WaveNet(
            batch_size=1,
            channels=256,
            dilations=[1],
            filter_width=2,
            residual_channels=256,
            dilation_channels=256,
        )

        x = np.linspace(-1, 1, 1000).astype(np.float32)

        # Test whether decoded signal is roughly equal to
        # what was decoded before
        with self.test_session() as sess:
            x1 = sess.run(net.decode(net.encode(x)))

        self.assertAllClose(x, x1, rtol=1e-1, atol=0.05)

        # Make sure that re-encoding leaves the waveform invariant
        with self.test_session() as sess:
            x2 = sess.run(net.decode(net.encode(x1)))

        self.assertAllClose(x1, x2)




