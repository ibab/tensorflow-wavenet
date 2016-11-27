"""Unit tests for the WaveNet that check that it can train on audio data."""

import json

import numpy as np
import sys
import tensorflow as tf
# import matplotlib.pyplot as plt
# import librosa

from wavenet import (WaveNetModel, time_to_batch, batch_to_time, causal_conv,
                     optimizer_factory, mu_law_decode)

SAMPLE_RATE_HZ = 2000.0  # Hz
TRAIN_ITERATIONS = 400
SAMPLE_DURATION = 0.5  # Seconds
SAMPLE_PERIOD_SECS = 1.0 / SAMPLE_RATE_HZ
MOMENTUM = 0.95
MOMENTUM_SCALAR_INPUT = 0.9
GENERATE_SAMPLES = 1000
QUANTIZATION_CHANNELS = 256
F1 = 155.56  # E-flat frequency in hz
F2 = 196.00  # G frequency in hz
F3 = 233.08  # B-flat frequency in hz


def make_sine_waves():
    """Creates a time-series of audio amplitudes corresponding to 3
    superimposed sine waves."""
    sample_period = 1.0/SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)

    amplitudes = (np.sin(times * 2.0 * np.pi * F1) / 3.0 +
                  np.sin(times * 2.0 * np.pi * F2) / 3.0 +
                  np.sin(times * 2.0 * np.pi * F3) / 3.0)

    return amplitudes


def generate_waveform(sess, net, fast_generation):
    samples = tf.placeholder(tf.int32)
    if fast_generation:
        next_sample_probs = net.predict_proba_incremental(samples)
        sess.run(net.init_ops)
        operations = [next_sample_probs]
        operations.extend(net.push_ops)
    else:
        next_sample_probs = net.predict_proba(samples)
        operations = [next_sample_probs]

    waveform = [128] * net.receptive_field
    decode = mu_law_decode(samples, QUANTIZATION_CHANNELS)
    if fast_generation:
        for sample in waveform[:-1]:
            sess.run(operations, feed_dict={samples: [sample]})

    for i in range(GENERATE_SAMPLES):
        if fast_generation:
            window = waveform[-1]
        else:
            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:]
            else:
                window = waveform

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(operations, feed_dict={samples: window})[0]
        sample = np.random.choice(
           np.arange(QUANTIZATION_CHANNELS), p=prediction)
        waveform.append(sample)
        # print("Generated {} of {}: {}".format(i, GENERATE_SAMPLES, sample))
        # sys.stdout.flush()

    # Skip the first number of samples equal to the size of the receptive
    # field minus one.
    waveform = np.array(waveform[net.receptive_field - 1:])
    decoded_waveform = sess.run(decode, feed_dict={samples: waveform})
    return decoded_waveform


def find_nearest(freqs, power_spectrum, frequency):
    # Return the power of the bin nearest to the target frequency.
    index = (np.abs(freqs - frequency)).argmin()
    return power_spectrum[index]


def check_waveform(assertion, generated_waveform):
    # librosa.output.write_wav('/tmp/sine_test.wav',
    #                          generated_waveform,
    #                          SAMPLE_RATE_HZ)
    power_spectrum = np.abs(np.fft.fft(generated_waveform))**2
    freqs = np.fft.fftfreq(generated_waveform.size, SAMPLE_PERIOD_SECS)
    indices = np.argsort(freqs)
    indices = [index for index in indices if freqs[index] >= 0 and
               freqs[index] <= 500.0]
    power_spectrum = power_spectrum[indices]
    freqs = freqs[indices]
    # plt.plot(freqs[indices], power_spectrum[indices])
    # plt.show()
    power_sum = np.sum(power_spectrum)
    f1_power = find_nearest(freqs, power_spectrum, F1)
    f2_power = find_nearest(freqs, power_spectrum, F2)
    f3_power = find_nearest(freqs, power_spectrum, F3)
    expected_power = f1_power + f2_power + f3_power
    # print("Power sum {}, F1 power:{}, F2 power:{}, F3 power:{}".
    #        format(power_sum, f1_power, f2_power, f3_power))

    # Expect most of the power to be at the 3 frequencies we trained
    # on.
    assertion(expected_power, 0.75 * power_sum)



class TestNet(tf.test.TestCase):
    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                skip_channels=32)
        self.optimizer_type = 'sgd'
        self.learning_rate = 0.02
        self.generate = False
        self.momentum = MOMENTUM

    # Train a net on a short clip of 3 sine waves superimposed
    # (an e-flat chord).
    #
    # Presumably it can overfit to such a simple signal. This test serves
    # as a smoke test where we just check that it runs end-to-end during
    # training, and learns this waveform.

    def testEndToEndTraining(self):
        audio = make_sine_waves()
        # Pad with 0s (silence) times size of the receptive field minus one,
        # because the first sample of the training data is 0 and if the network
        # learns to predict silence based on silence, it will generate only
        # silence.
        audio = np.pad(audio, [self.net.receptive_field - 1, 0], 'constant')
        np.random.seed(42)

        # if self.generate:
        #    librosa.output.write_wav('/tmp/sine_train.wav', audio,
        #                             SAMPLE_RATE_HZ)
        #    power_spectrum = np.abs(np.fft.fft(audio))**2
        #    freqs = np.fft.fftfreq(audio.size, SAMPLE_PERIOD_SECS)
        #    indices = np.argsort(freqs)
        #    indices = [index for index in indices if freqs[index] >= 0 and
        #                                             freqs[index] <= 500.0]
        #    plt.plot(freqs[indices], power_spectrum[indices])
        #    plt.show()

        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        loss = self.net.loss(audio_tensor)
        optimizer = optimizer_factory[self.optimizer_type](
                      learning_rate=self.learning_rate, momentum=self.momentum)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)
        init = tf.initialize_all_variables()

        generated_waveform = None
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

            # saver = tf.train.Saver(var_list=tf.trainable_variables())
            # saver.save(sess, '/tmp/sine_test_model.ckpt', global_step=i)
            if self.generate:
                # Check non-incremental generation
                generated_waveform = generate_waveform(sess, self.net, False)
                check_waveform(self.assertGreater, generated_waveform)

                # Check incremental generation
                generated_waveform = generate_waveform(sess, self.net, True)
                check_waveform(self.assertGreater, generated_waveform)


class TestNetWithBiases(TestNet):

    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                use_biases=True,
                                skip_channels=32)
        self.optimizer_type = 'sgd'
        self.learning_rate = 0.02
        self.generate = False
        self.momentum = MOMENTUM


class TestNetWithRMSProp(TestNet):

    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                skip_channels=256)
        self.optimizer_type = 'rmsprop'
        self.learning_rate = 0.001
        self.generate = True
        self.momentum = MOMENTUM


class TestNetWithScalarInput(TestNet):

    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                use_biases=True,
                                skip_channels=32,
                                scalar_input=True,
                                initial_filter_width=4)
        self.optimizer_type = 'rmsprop'
        self.learning_rate = 0.001
        self.generate = False
        self.momentum = MOMENTUM_SCALAR_INPUT


if __name__ == '__main__':
    tf.test.main()
