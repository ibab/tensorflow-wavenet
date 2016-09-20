import tensorflow as tf

from wavenet import WaveNet
from test_encoding import manual_encode
import json
import numpy as np

def manual_decode(signal, channels):
    # Calculate inverse mu-law companding and dequantization
    mu = channels
    output = signal.astype(np.float32)
    y = (2*output - 1) / mu
    x = np.sign(y) * (np.exp(y * np.log(1 + mu)) - 1) / mu

    return x


class TestDecode(tf.test.TestCase):

    def setUp(self):
        quantization_steps=256
        self.net = WaveNet(batch_size = 1,
                           channels = quantization_steps,
                           dilations= [1, 2, 4, 8, 16, 32, 64, 128, 256,
                                       1, 2, 4, 8, 16, 32, 64, 128, 256],
                           filter_width=2,
                           residual_channels = 16,
                           dilation_channels = 16)


    def testUniformRandomNoise(self):
        # Input: uniformly distributed random noise
        with self.test_session():
            # Audio encoding parameters
            number_of_samples = 512
            mu = self.net.channels

            # Random seed for repeatability of test
            np.random.seed(40)

            # Generate audio input for one channel
            audio = np.random.uniform(-1, 1, number_of_samples)

            # Encode and decode audio input 
            encoded_audio_manual = manual_encode(audio, mu)
            decoded_audio_manual = manual_decode(encoded_audio_manual, mu)

            decoded_audio_net_tensor = self.net.decode(encoded_audio_manual)
            decoded_audio_net = decoded_audio_net_tensor.eval()

            # Shapes should be the same
            self.assertAllEqual(decoded_audio_net.shape, decoded_audio_manual.shape)

            # Outputs should be within some rounding errors
            self.assertAllClose(decoded_audio_net, decoded_audio_manual, rtol=1e-7)

    def testRandomConstant(self):
        # Input: random constant
        with self.test_session():
            # Audio encoding parameters
            number_of_samples = 2048
            mu = self.net.channels

            # Random seed for repeatability of test
            np.random.seed(42)

            # Generate audio for one channel
            audio = np.zeros(number_of_samples)
            audio.fill(np.random.uniform(-1, 1))

            encoded_audio_manual = manual_encode(audio, mu)

            decoded_audio_manual = manual_decode(encoded_audio_manual, mu)
            decoded_audio_net_tensor = self.net.decode(encoded_audio_manual)
            decoded_audio_net = decoded_audio_net_tensor.eval()

            # Shapes should be the same
            self.assertAllEqual(decoded_audio_net.shape, decoded_audio_manual.shape)

            # Outputs should be within some rounding errors
            self.assertAllClose(decoded_audio_net, decoded_audio_manual, rtol=1e-7)

    def testRamp(self):
        # Input: ramp [-1, 1[ 
        with self.test_session():
            # Audio encoding parameters
            number_of_samples = 256
            mu = self.net.channels

            # Original audio input is a ramp from -1 to 1
            number_of_steps = 2 / number_of_samples
            audio = np.arange(-1.0, 1.0, number_of_steps)

            encoded_audio_manual = manual_encode(audio, number_of_samples)

            decoded_audio_manual = manual_decode(encoded_audio_manual, number_of_samples)
            decoded_audio_net_tensor = self.net.decode(encoded_audio_manual)
            decoded_audio_net = decoded_audio_net_tensor.eval()

            # Shapes should be the same
            self.assertAllEqual(decoded_audio_net.shape, decoded_audio_manual.shape)

            # Outputs should be within some rounding errors
            self.assertAllClose(decoded_audio_net, decoded_audio_manual, rtol=1e-7)

    def testZeros(self):
        # Input: zeros
        with self.test_session():
            # Audio encoding parameters
            number_of_samples = 4096
            mu = self.net.channels

            # Some audio input
            audio = np.zeros(number_of_samples)
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

            encoded_audio_manual = manual_encode(audio, mu)

            decoded_audio_manual = manual_decode(encoded_audio_manual, mu)
            decoded_audio_net_tensor = self.net.decode(encoded_audio_manual)
            decoded_audio_net = decoded_audio_net_tensor.eval()

            # Shapes should be the same
            self.assertAllEqual(decoded_audio_net.shape, decoded_audio_manual.shape)

            # Outputs should be within some rounding errors
            self.assertAllClose(decoded_audio_net, decoded_audio_manual, rtol=1e-7)

    def testNegativeDilation(self):
        # Input: manual data
        # Test if code throws a TypeError when channel size is negative
        with self.test_session():
            self.net.channels = -1

            # Some encoded data
            encoded_audio_manual = [0, 255, 243, 31, 156, 229, 0, 235, 202, 18]
            decoded_audio_net_tensor = self.net.decode(encoded_audio_manual)
            decoded_audio_net = decoded_audio_net_tensor.eval()

            self.assertRaises(TypeError, self.net.decode(encoded_audio_manual))


if __name__ == '__main__':
    tf.test.main()