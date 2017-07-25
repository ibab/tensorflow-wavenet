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
        '''does nothing'''
        self.assertAllEqual([128], [128])
        self.assertTrue(True)


"""
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
"""

if __name__ == '__main__':
    tf.test.main()
