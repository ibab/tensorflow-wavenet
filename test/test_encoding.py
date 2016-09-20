import tensorflow as tf

from wavenet import WaveNet
import json
import numpy as np

def manual_encode(signal, channels):
    # Manual mu-law companding and mu-bits quantization
    mu = channels - 1

    magnitude = np.log(1 + mu * np.abs(signal)) / np.log(1. + mu)
    signal = np.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    signal = (signal + 1) / 2 * mu
    quantized_signal = signal.astype(np.int32)

    return quantized_signal


class TestEncoding(tf.test.TestCase):

    def setUp(self):
        quantization_steps=256
        self.net = WaveNet(batch_size = 1,
                           channels = quantization_steps,
                           dilations= [1, 2, 4, 8, 16, 32, 64, 128, 256,
                                       1, 2, 4, 8, 16, 32, 64, 128, 256],
                           filter_width=2,
                           residual_channels = 16,
                           dilation_channels = 16)

    def testPrecomputed(self):
        # Input: signals encoded by hand
        with self.test_session():
            # Audio encoding parameters
            number_of_samples = 10
            mu = self.net.channels
            audio = [-1.0, 1.0, 0.6, -0.25, 0.01, 0.33, -0.9999, 0.42, 0.1, -0.45]
            encoded_audio_manual = [0, 255, 243, 31, 156, 229, 0, 235, 202, 18]

            # Tensor processing
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            preprocessed_audio_tensor = self.net._preprocess(audio_tensor)

            encoded_audio_WaveNet = preprocessed_audio_tensor.eval()
            self.assertAllEqual(encoded_audio_WaveNet, encoded_audio_manual)
    
    def testUniformRandomNoise(self):
        # Input: uniform random noise 
        with self.test_session():
            # Random seed for repeatability of test
            np.random.seed(42)

            # Audio encoding parameters
            number_of_samples = 1024
            mu = self.net.channels
            audio = np.random.uniform(-1, 1, number_of_samples)

            # Tensor processing
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            preprocessed_audio_tensor = self.net._preprocess(audio_tensor)

            encoded_audio_WaveNet = preprocessed_audio_tensor.eval()
            encoded_audio_manual = manual_encode(audio, mu)
            self.assertAllEqual(encoded_audio_WaveNet, encoded_audio_manual)

    def testRandomConstant(self):
        # Input: random constant
        with self.test_session():
            # Random seed for repeatability of test
            np.random.seed(420)

            # Audio encoding parameters
            number_of_samples = 1024
            mu = self.net.channels
            audio = np.zeros(number_of_samples)
            audio.fill(np.random.uniform(-1, 1))

            # Tensor processing
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            preprocessed_audio_tensor = self.net._preprocess(audio_tensor)

            encoded_audio_WaveNet = preprocessed_audio_tensor.eval()
            encoded_audio_manual = manual_encode(audio, mu)
            self.assertAllEqual(encoded_audio_WaveNet, encoded_audio_manual)

    def testRamp(self):
        # Input: ramp [-1, 1[ 
        with self.test_session():
            # Audio encoding parameters
            number_of_samples = 1024
            mu = self.net.channels
            number_of_steps = 2 / number_of_samples
            audio = np.arange(-1.0, 1.0, number_of_steps)

            # Tensor processing
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            preprocessed_audio_tensor = self.net._preprocess(audio_tensor)

            encoded_audio_WaveNet = preprocessed_audio_tensor.eval()
            encoded_audio_manual = manual_encode(audio, mu)
            self.assertAllEqual(encoded_audio_WaveNet, encoded_audio_manual)

    def testZeros(self):
        # Input: zeros
        with self.test_session():
            # Audio encoding parameters
            number_of_samples = 1024
            mu = self.net.channels
            audio = np.zeros(number_of_samples)

            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            preprocessed_audio_tensor = self.net._preprocess(audio_tensor)

            encoded_audio_WaveNet = preprocessed_audio_tensor.eval()
            encoded_audio_manual = manual_encode(audio, mu)
            self.assertAllEqual(encoded_audio_WaveNet, encoded_audio_manual)

    def testNegativeDilation(self):
        # Input: zeros
        # Test if code throws a TypeError when channel size is negative
        with self.test_session():
            # Random seed for repeatability of test
            np.random.seed(11)

            # Audio encoding parameters
            self.net.channels = -1
            number_of_samples = 1024
            mu = self.net.channels
            audio = np.zeros(number_of_samples)
            audio.fill(np.random.uniform(-1, 1))

            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            preprocessed_audio_tensor = self.net._preprocess(audio_tensor)

            self.assertRaises(TypeError, preprocessed_audio_tensor.eval())


if __name__ == '__main__':
    tf.test.main() 