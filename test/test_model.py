"""Unit tests for the WaveNet that check that it can train on audio data."""
import json
import numpy as np
import sys
import tensorflow as tf
import random
import os

from wavenet import (WaveNetModel, time_to_batch, batch_to_time, causal_conv,
                     optimizer_factory, mu_law_decode)

SAMPLE_RATE_HZ = 2000.0  # Hz
TRAIN_ITERATIONS = 400
SAMPLE_DURATION = 0.5  # Seconds
SAMPLE_PERIOD_SECS = 1.0 / SAMPLE_RATE_HZ
MOMENTUM = 0.95
GENERATE_SAMPLES = 1000
QUANTIZATION_CHANNELS = 256
NUM_SPEAKERS = 3
F1 = 155.56  # E-flat frequency in hz
F2 = 196.00  # G frequency in hz
F3 = 233.08  # B-flat frequency in hz


def make_sine_waves(global_conditioning):
    """Creates a time-series of sinusoidal audio amplitudes."""
    sample_period = 1.0/SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)

    if global_conditioning:
        LEADING_SILENCE = random.randint(10, 128)
        amplitudes = np.zeros(shape=(NUM_SPEAKERS, len(times)))
        amplitudes[0, 0:LEADING_SILENCE] = 0.0
        amplitudes[1, 0:LEADING_SILENCE] = 0.0
        amplitudes[2, 0:LEADING_SILENCE] = 0.0
        start_time = LEADING_SILENCE / SAMPLE_RATE_HZ
        times = times[LEADING_SILENCE:] - start_time
        amplitudes[0, LEADING_SILENCE:] = 0.6 * np.sin(times *
                                                       2.0 * np.pi * F1)
        amplitudes[1, LEADING_SILENCE:] = 0.5 * np.sin(times *
                                                       2.0 * np.pi * F2)
        amplitudes[2, LEADING_SILENCE:] = 0.4 * np.sin(times *
                                                       2.0 * np.pi * F3)
        speaker_ids = np.zeros((NUM_SPEAKERS, 1), dtype=np.int)
        speaker_ids[0, 0] = 0
        speaker_ids[1, 0] = 1
        speaker_ids[2, 0] = 2
    else:
        amplitudes = (np.sin(times * 2.0 * np.pi * F1) / 3.0 +
                      np.sin(times * 2.0 * np.pi * F2) / 3.0 +
                      np.sin(times * 2.0 * np.pi * F3) / 3.0)
        speaker_ids = None

    return amplitudes, speaker_ids


def generate_waveform(sess, net, fast_generation, gc, samples_placeholder,
                      gc_placeholder, operations):
    waveform = [128] * net.receptive_field
    if fast_generation:
        for sample in waveform[:-1]:
            sess.run(operations, feed_dict={samples_placeholder: [sample]})

    for i in range(GENERATE_SAMPLES):
        if i % 100 == 0:
            print("Generating {} of {}.".format(i, GENERATE_SAMPLES))
            sys.stdout.flush()
        if fast_generation:
            window = waveform[-1]
        else:
            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:]
            else:
                window = waveform

        # Run the WaveNet to predict the next sample.
        feed_dict = {samples_placeholder: window}
        if gc is not None:
            feed_dict[gc_placeholder] = gc
        results = sess.run(operations, feed_dict=feed_dict)

        sample = np.random.choice(
           np.arange(QUANTIZATION_CHANNELS), p=results[0])
        waveform.append(sample)

    # Skip the first number of samples equal to the size of the receptive
    # field minus one.
    waveform = np.array(waveform[net.receptive_field - 1:])
    decode = mu_law_decode(samples_placeholder, QUANTIZATION_CHANNELS)
    decoded_waveform = sess.run(decode,
                                feed_dict={samples_placeholder: waveform})
    return decoded_waveform


def generate_waveforms(sess, net, fast_generation, global_condition):
    samples_placeholder = tf.placeholder(tf.int32)
    gc_placeholder = tf.placeholder(tf.int32) if global_condition is not None \
        else None

    net.batch_size = 1

    if fast_generation:
        next_sample_probs = net.predict_proba_incremental(samples_placeholder,
                                                          global_condition)
        sess.run(net.init_ops)
        operations = [next_sample_probs]
        operations.extend(net.push_ops)
    else:
        next_sample_probs = net.predict_proba(samples_placeholder,
                                              gc_placeholder)
        operations = [next_sample_probs]

    num_waveforms = 1 if global_condition is None else  \
        global_condition.shape[0]
    gc = None
    waveforms = [None] * num_waveforms
    for waveform_index in range(num_waveforms):
        if global_condition is not None:
            gc = global_condition[waveform_index, :]
        # Generate a waveform for each speaker id.
        print("Generating waveform {}.".format(waveform_index))
        waveforms[waveform_index] = generate_waveform(
            sess, net, fast_generation, gc, samples_placeholder,
            gc_placeholder, operations)

    return waveforms, global_condition


def find_nearest(freqs, power_spectrum, frequency):
    # Return the power of the bin nearest to the target frequency.
    index = (np.abs(freqs - frequency)).argmin()
    return power_spectrum[index]


def check_waveform(assertion, generated_waveform, gc_category):
    # librosa.output.write_wav('/tmp/sine_test{}.wav'.format(gc_category),
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
    if gc_category is None:
        # We are not globally conditioning to select one of the three sine
        # waves, so expect it across all three.
        expected_power = f1_power + f2_power + f3_power
        assertion(expected_power, 0.7 * power_sum)
    else:
        # We expect spectral power at the selected frequency
        # corresponding to the gc_category to be much higher than at the other
        # two frequencies.
        frequency_lut = {0: f1_power, 1: f2_power, 2: f3_power}
        other_freqs_lut = {0: f2_power + f3_power,
                           1: f1_power + f3_power,
                           2: f1_power + f2_power}
        expected_power = frequency_lut[gc_category]
        # Power at the selected frequency should be at least 10 times greater
        # than at other frequences.
        # This is a weak criterion, but still detects implementation errors
        # in the code.
        assertion(expected_power, 10.0*other_freqs_lut[gc_category])


class TestNet(tf.test.TestCase):
    def setUp(self):
        print('TestNet setup.')
        sys.stdout.flush()

        self.optimizer_type = 'sgd'
        self.learning_rate = 0.02
        self.generate = False
        self.momentum = MOMENTUM
        self.global_conditioning = False
        self.train_iters = TRAIN_ITERATIONS
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                skip_channels=32,
                                global_condition_channels=None,
                                global_condition_cardinality=None)

    def _save_net(sess):
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.save(sess, os.path.join('tmp', 'test.ckpt'))

    # Train a net on a short clip of 3 sine waves superimposed
    # (an e-flat chord).
    #
    # Presumably it can overfit to such a simple signal. This test serves
    # as a smoke test where we just check that it runs end-to-end during
    # training, and learns this waveform.

    def testEndToEndTraining(self):
        def CreateTrainingFeedDict(audio, speaker_ids, audio_placeholder,
                                   gc_placeholder, i):
            speaker_index = 0
            if speaker_ids is None:
                # No global conditioning.
                feed_dict = {audio_placeholder: audio}
            else:
                feed_dict = {audio_placeholder: audio,
                             gc_placeholder: speaker_ids}
            return feed_dict, speaker_index

        np.random.seed(42)
        audio, speaker_ids = make_sine_waves(self.global_conditioning)
        # Pad with 0s (silence) times size of the receptive field minus one,
        # because the first sample of the training data is 0 and if the network
        # learns to predict silence based on silence, it will generate only
        # silence.
        if self.global_conditioning:
            audio = np.pad(audio, ((0, 0), (self.net.receptive_field - 1, 0)),
                           'constant')
        else:
            audio = np.pad(audio, (self.net.receptive_field - 1, 0),
                           'constant')

        audio_placeholder = tf.placeholder(dtype=tf.float32)
        gc_placeholder = tf.placeholder(dtype=tf.int32)  \
            if self.global_conditioning else None

        loss = self.net.loss(input_batch=audio_placeholder,
                             global_condition_batch=gc_placeholder)
        optimizer = optimizer_factory[self.optimizer_type](
                      learning_rate=self.learning_rate, momentum=self.momentum)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)
        init = tf.initialize_all_variables()

        generated_waveform = None
        max_allowed_loss = 0.1
        loss_val = max_allowed_loss
        initial_loss = None
        operations = [loss, optim]
        with self.test_session() as sess:
            feed_dict, speaker_index = CreateTrainingFeedDict(
                audio, speaker_ids, audio_placeholder, gc_placeholder, 0)
            sess.run(init)
            initial_loss = sess.run(loss, feed_dict=feed_dict)
            for i in range(self.train_iters):
                feed_dict, speaker_index = CreateTrainingFeedDict(
                    audio, speaker_ids, audio_placeholder, gc_placeholder, i)
                [results] = sess.run([operations], feed_dict=feed_dict)
                if i % 100 == 0:
                    print("i: %d loss: %f" % (i, results[0]))

            loss_val = results[0]

            # Sanity check the initial loss was larger.
            self.assertGreater(initial_loss, max_allowed_loss)

            # Loss after training should be small.
            self.assertLess(loss_val, max_allowed_loss)

            # Loss should be at least two orders of magnitude better
            # than before training.
            self.assertLess(loss_val / initial_loss, 0.02)

            if self.generate:
                # self._save_net(sess)
                if self.global_conditioning:
                    # Check non-fast-generated waveform.
                    generated_waveforms, ids = generate_waveforms(
                        sess, self.net, False, speaker_ids)
                    for (waveform, id) in zip(generated_waveforms, ids):
                        check_waveform(self.assertGreater, waveform, id[0])

                    # Check fast-generated wveform.
                    # generated_waveforms, ids = generate_waveforms(sess,
                    #     self.net, True, speaker_ids)
                    # for (waveform, id) in zip(generated_waveforms, ids):
                    #     print("Checking fast wf for id{}".format(id[0]))
                    #     check_waveform( self.assertGreater, waveform, id[0])

                else:
                    # Check non-incremental generation
                    generated_waveforms, _ = generate_waveforms(
                        sess, self.net, False, None)
                    check_waveform(
                        self.assertGreater, generated_waveforms[0], None)
                    # Check incremental generation
                    generated_waveform = generate_waveforms(
                        sess, self.net, True, None)
                    check_waveform(
                        self.assertGreater, generated_waveforms[0], None)


class TestNetWithBiases(TestNet):

    def setUp(self):
        print('TestNetWithBias setup.')
        sys.stdout.flush()

        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                use_biases=True,
                                skip_channels=32)
        self.optimizer_type = 'sgd'
        self.learning_rate = 0.02
        self.generate = False
        self.momentum = MOMENTUM
        self.global_conditioning = False
        self.train_iters = TRAIN_ITERATIONS


class TestNetWithRMSProp(TestNet):

    def setUp(self):
        print('TestNetWithRMSProp setup.')
        sys.stdout.flush()

        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                skip_channels=256)
        self.optimizer_type = 'rmsprop'
        self.learning_rate = 0.001
        self.generate = True
        self.momentum = MOMENTUM
        self.train_iters = TRAIN_ITERATIONS
        self.global_conditioning = False


class TestNetWithScalarInput(TestNet):

    def setUp(self):
        print('TestNetWithScalarInput setup.')
        sys.stdout.flush()

        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                use_biases=True,
                                skip_channels=32,
                                scalar_input=True,
                                initial_filter_width=4)
        self.optimizer_type = 'sgd'
        self.learning_rate = 0.01
        self.generate = False
        self.momentum = MOMENTUM
        self.global_conditioning = False
        self.train_iters = 1000


class TestNetWithGlobalConditioning(TestNet):
    def setUp(self):
        print('TestNetWithGlobalConditioning setup.')
        sys.stdout.flush()

        self.optimizer_type = 'sgd'
        self.learning_rate = 0.01
        self.generate = True
        self.momentum = MOMENTUM
        self.global_conditioning = True
        self.train_iters = 1000
        self.net = WaveNetModel(batch_size=NUM_SPEAKERS,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=QUANTIZATION_CHANNELS,
                                use_biases=True,
                                skip_channels=256,
                                global_condition_channels=NUM_SPEAKERS,
                                global_condition_cardinality=NUM_SPEAKERS)


if __name__ == '__main__':
    tf.test.main()
