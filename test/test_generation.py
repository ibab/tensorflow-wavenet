import numpy as np

import tensorflow as tf

from wavenet import WaveNetModel


class TestGeneration(tf.test.TestCase):

    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                                filter_width=2,
                                residual_channels=16,
                                dilation_channels=16,
                                quantization_channels=128,
                                skip_channels=32)

    def testGenerateSimple(self):
        '''Generate a few samples using the naive method and
        perform sanity checks on the output.'''
        waveform = tf.placeholder(tf.int32)
        np.random.seed(0)
        data = np.random.randint(128, size=1000)
        proba = self.net.predict_proba(waveform)

        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            proba = sess.run(proba, feed_dict={waveform: data})

        self.assertAllEqual(proba.shape, [128])
        self.assertTrue(np.all((proba >= 0) & (proba <= (128 - 1))))

    def testGenerateFast(self):
        '''Generate a few samples using the fast method and
        perform sanity checks on the output.'''
        waveform = tf.placeholder(tf.int32)
        np.random.seed(0)
        data = np.random.randint(128)
        proba = self.net.predict_proba_incremental(waveform)

        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(self.net.init_ops)
            proba = sess.run(proba, feed_dict={waveform: data})

        self.assertAllEqual(proba.shape, [128])
        self.assertTrue(np.all((proba >= 0) & (proba <= (128 - 1))))

    def testCompareSimpleFast(self):
        waveform = tf.placeholder(tf.int32)
        np.random.seed(0)
        data = np.random.randint(128, size=1000)
        proba = self.net.predict_proba(waveform)
        proba_fast = self.net.predict_proba_incremental(waveform)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(self.net.init_ops)
            # Prime the incremental generation with all samples
            # except the last one
            for x in data[:-1]:
                proba_fast_ = sess.run(
                    [proba_fast, self.net.push_ops],
                    feed_dict={waveform: x})

            # Get the last sample from the incremental generator
            proba_fast_ = sess.run(
                proba_fast,
                feed_dict={waveform: data[-1]})
            # Get the sample from the simple generator
            proba_ = sess.run(proba, feed_dict={waveform: data})
            self.assertAllClose(proba_, proba_fast_)


class TestGenerationBiases(TestGeneration):

    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                                filter_width=2,
                                use_biases=True,
                                residual_channels=16,
                                dilation_channels=16,
                                quantization_channels=128,
                                skip_channels=32)


if __name__ == '__main__':
    tf.test.main()
