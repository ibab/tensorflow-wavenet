"""Unit tests for the WaveNet that check that it can train on audio data."""

import json

import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, time_to_batch, batch_to_time, causal_conv

SAMPLE_RATE_HZ = 2000.0  # Hz
TRAIN_ITERATIONS = 400
LEARN_RATE = 0.02
SAMPLE_DURATION = 0.2  # Seconds
MOMENTUM = 0.9


def MakeSineWaves():
    """Creates a time-series of audio amplitudes corresponding to 3
    superimposed sine waves."""
    # Frequencies of the sine waves in Hz.
    f1 = 155.56  # E-flat
    f2 = 196.00  # G
    f3 = 233.08  # B-flat
    sample_period = 1.0/SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)

    amplitudes = (np.sin(times * 2.0 * np.pi * f1) / 3.0 +
                  np.cos(times * 2.0 * np.pi * f2) / 3.0 +
                  np.sin(times * 2.0 * np.pi * f3) / 3.0)

    return amplitudes


class TestNet(tf.test.TestCase):
    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256,
                                           1, 2, 4, 8, 16, 32, 64, 128, 256],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                skip_channels=32)

    # Train a net on a short clip of 3 sine waves superimposed
    # (an e-flat chord).
    #
    # Presumably it can overfit to such a simple signal. This test serves
    # as a smoke test where we just check that it runs end-to-end during
    # training, and learns this waveform.

    def testEndToEndTraining(self):
        audio = MakeSineWaves()
        np.random.seed(42)

        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        loss = self.net.loss(audio_tensor)
        optimizer = tf.train.MomentumOptimizer(learning_rate=LEARN_RATE,
                                               momentum=MOMENTUM)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)
        init = tf.initialize_all_variables()

        max_allowed_loss = 0.1
        loss_val = max_allowed_loss
        initial_loss = None
        with self.test_session() as sess:
            sess.run(init)
            initial_loss = sess.run(loss)
            for i in range(TRAIN_ITERATIONS):
                loss_val, _ = sess.run([loss, optim])
                # if i % 10 == 0:
                #     print("i: %d loss: %f" % (i, loss_val))

        # Sanity check the initial loss was larger.
        self.assertGreater(initial_loss, max_allowed_loss)

        # Loss after training should be small.
        self.assertLess(loss_val, max_allowed_loss)

        # Loss should be at least two orders of magnitude better
        # than before training.
        self.assertLess(loss_val / initial_loss, 0.01)


class TestNetWithBiases(TestNet):

    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256,
                                           1, 2, 4, 8, 16, 32, 64, 128, 256],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                use_biases=True,
                                skip_channels=32)


class TestNetWithScalarInput(TestNet):

    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256,
                                           1, 2, 4, 8, 16, 32, 64, 128, 256],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                use_biases=True,
                                skip_channels=32,
                                scalar_input=True,
                                initial_filter_width=32)

if __name__ == '__main__':
    tf.test.main()
