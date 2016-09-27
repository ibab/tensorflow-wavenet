from __future__ import division

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNet
from wavenet_ops import mu_law_decode, mu_law_encode

SAMPLES = 16000
LOGDIR = './logdir'
WINDOW = 8000
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        default=True,
        type=_str_to_bool,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    return parser.parse_args()


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

def create_seed(filename, sample_rate, quantization_channels, window_size=WINDOW):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.size(quantized) + tf.constant(window_size) - tf.constant(1)

    return quantized[:cut_index]

def main():
    args = get_arguments()
    logdir = os.path.join(args.logdir, 'train', str(datetime.now()))
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNet(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        fast_generation=args.fast_generation)

    samples = tf.placeholder(tf.int32)

    next_sample = net.predict_proba(samples)

    if args.fast_generation:
        sess.run(tf.initialize_all_variables())
        sess.run(net.init_ops)

    variables_to_restore = {var.name[:-2]: var for var in tf.all_variables(
    ) if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

    quantization_channels = wavenet_params['quantization_channels']
    if args.wav_seed:
        seed = create_seed(args.wav_seed,
                    wavenet_params['sample_rate'],
                    quantization_channels)
        waveform = sess.run(seed).tolist()
    else:
        waveform = np.random.randint(quantization_channels, size=(1,)).tolist()

    for step in range(args.samples):
        if args.fast_generation:
            window = waveform[-1]
            outputs = [next_sample]
            outputs.extend(net.push_ops)
        else:
            if len(waveform) > args.window:
                window = waveform[-args.window:]
            else:
                window = waveform
            outputs = [next_sample]

        prediction = sess.run(outputs, feed_dict={samples: window})[0]
        sample = np.random.choice(
            np.arange(quantization_channels), p=prediction)
        waveform.append(sample)
        print('Sample {:3<d}/{:3<d}: {}'
              .format(step + 1, args.samples, sample))

        if (args.wav_out_path and args.save_every and
                (step + 1) % args.save_every == 0):

            out = sess.run(decode, feed_dict={samples: waveform})
            write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.train.SummaryWriter(
        os.path.join(logdir, 'generation', datestring))
    tf.audio_summary('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.merge_all_summaries()

    summary_out = sess.run(summaries,
                           feed_dict={
                               samples: np.reshape(waveform, [-1, 1])
                           })
    writer.add_summary(summary_out)

    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
