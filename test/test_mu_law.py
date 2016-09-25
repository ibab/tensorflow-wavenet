import numpy as np
import tensorflow as tf

from wavenet_ops import mu_law_encode, mu_law_decode


# A set of mu law encode/decode functions implemented
# in numpy
def manual_mu_law_encode(signal, quantization_channels):
    # Manual mu-law companding and mu-bits quantization
    mu = quantization_channels - 1

    magnitude = np.log(1 + mu * np.abs(signal)) / np.log(1. + mu)
    signal = np.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    signal = (signal + 1) / 2 * mu + 0.5
    quantized_signal = signal.astype(np.int32)

    return quantized_signal


def manual_mu_law_decode(signal, quantization_channels):
    # Calculate inverse mu-law companding and dequantization
    mu = quantization_channels - 1
    y = signal.astype(np.float32)

    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu)**abs(y) - 1.0)
    return x


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

    def testEncodePrecomputed(self):
        channels = 256
        number_of_samples = 10
        x = np.array([-1.0, 1.0, 0.6, -0.25, 0.01,
                      0.33, -0.9999, 0.42, 0.1, -0.45]).astype(np.float32)
        encoded_manual = np.array([0, 255, 243, 32, 157,
                                   230, 0, 235, 203, 18]).astype(np.int32)

        with self.test_session() as sess:
            encoded = sess.run(mu_law_encode(x, channels))

        self.assertAllEqual(encoded_manual, encoded)

    def testEncodeUniformRandomNoise(self):
        np.random.seed(42)  # For repeatability of test.

        channels = 256
        number_of_samples = 2048
        x = np.random.uniform(-1, 1, number_of_samples).astype(np.float32)
        manual_encode = manual_mu_law_encode(x, channels)

        with self.test_session() as sess:
            encode = sess.run(mu_law_encode(x, channels))

        self.assertAllEqual(manual_encode, encode)

    def testEncodeRandomConstant(self):
        np.random.seed(1944)  # For repeatability of test.

        channels = 256
        number_of_samples = 1024
        x = np.zeros(number_of_samples).astype(np.float32)
        x.fill(np.random.uniform(-1, 1))
        manual_encode = manual_mu_law_encode(x, channels)

        with self.test_session() as sess:
            encode = sess.run(mu_law_encode(x, channels))

        self.assertAllEqual(manual_encode, encode)

    def testEncodeRamp(self):
        np.random.seed(1944)  # For repeatability of test.

        channels = 256
        number_of_samples = 1024
        number_of_steps = 2.0 / number_of_samples
        x = np.arange(-1.0, 1.0, number_of_steps).astype(np.float32)
        manual_encode = manual_mu_law_encode(x, channels)

        with self.test_session() as sess:
            encode = sess.run(mu_law_encode(x, channels))

        self.assertAllEqual(manual_encode, encode)

    def testEncodeZeros(self):
        np.random.seed(1944)  # For repeatability of test.

        channels = 256
        number_of_samples = 1024
        x = np.zeros(number_of_samples).astype(np.float32)
        manual_encode = manual_mu_law_encode(x, channels)

        with self.test_session() as sess:
            encode = sess.run(mu_law_encode(x, channels))

        self.assertAllEqual(manual_encode, encode)

    def testEncodeNegativeChannelSize(self):
        np.random.seed(1944)  # For repeatability of test.

        channels = -256
        number_of_samples = 1024
        x = np.zeros(number_of_samples).astype(np.float32)
        manual_encode = manual_mu_law_encode(x, channels)

        with self.test_session() as sess:
            self.assertRaises(TypeError, sess.run(mu_law_encode(x, channels)))

    def testDecodeUniformRandomNoise(self):
        np.random.seed(1944)  # For repeatability of test.

        channels = 256
        number_of_samples = 10
        x = np.random.uniform(-1, 1, number_of_samples).astype(np.float32)
        y = manual_mu_law_encode(x, channels)
        manual_decode = manual_mu_law_decode(y, channels)

        with self.test_session() as sess:
            decode = sess.run(mu_law_decode(y, channels))

        self.assertAllEqual(manual_decode, decode)

    def testDecodeUniformRandomNoise(self):
        np.random.seed(40)

        channels = 128
        number_of_samples = 512
        x = np.random.uniform(-1, 1, number_of_samples)
        y = manual_mu_law_encode(x, channels)
        decoded_manual = manual_mu_law_decode(y, channels)

        with self.test_session() as sess:
            decode = sess.run(mu_law_decode(y, channels))

        self.assertAllEqual(decoded_manual, decode)

    def testDecodeRandomConstant(self):
        np.random.seed(40)

        channels = 128
        number_of_samples = 512
        x = np.zeros(number_of_samples)
        x.fill(np.random.uniform(-1, 1))
        y = manual_mu_law_encode(x, channels)
        decoded_manual = manual_mu_law_decode(y, channels)

        with self.test_session() as sess:
            decode = sess.run(mu_law_decode(y, channels))

        self.assertAllEqual(decoded_manual, decode)

    def testDecodeRamp(self):
        np.random.seed(40)

        channels = 128
        number_of_samples = 512
        number_of_steps = 2.0 / number_of_samples
        x = np.arange(-1.0, 1.0, number_of_steps)
        y = manual_mu_law_encode(x, channels)
        decoded_manual = manual_mu_law_decode(y, channels)

        with self.test_session() as sess:
            decode = sess.run(mu_law_decode(y, channels))

        self.assertAllEqual(decoded_manual, decode)

    def testDecodeZeros(self):
        np.random.seed(40)

        channels = 128
        number_of_samples = 100
        x = np.zeros(number_of_samples)
        y = manual_mu_law_encode(x, channels)
        decoded_manual = manual_mu_law_decode(y, channels)

        with self.test_session() as sess:
            decode = sess.run(mu_law_decode(y, channels))

        self.assertAllEqual(decoded_manual, decode)

    def testDecodeNegativeDilation(self):
        channels = 10
        y = [0, 255, 243, 31, 156, 229, 0, 235, 202, 18]

        with self.test_session() as sess:
            self.assertRaises(TypeError, sess.run(mu_law_decode(y, channels)))


if __name__ == '__main__':
    tf.test.main()
