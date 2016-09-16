
import argparse
from datetime import datetime
import json
import os

import numpy as np
import tensorflow as tf

from wavenet import WaveNet

SAMPLES = 16000
LOGDIR = './logdir'
WINDOW = 256
WAVENET_PARAMS = './wavenet_params.json'

def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument('checkpoint', type=str,
                        help='Which model checkpoint to generate from')
    parser.add_argument('--samples', type=int, default=SAMPLES,
                        help='How many waveform samples to generate')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Directory in which to store the logging '
                        'information for TensorBoard.')
    parser.add_argument('--window', type=int, default=WINDOW,
                        help='The number of past samples to take into '
                        'account at each step')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters')
    parser.add_argument('--wav_out_path', type=str, default=None,
                        help='Path to output wav file')
    return parser.parse_args()

def main():
    args = get_arguments()
    logdir = os.path.join(args.logdir, 'train', str(datetime.now()))
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNet(
        1,
        wavenet_params['quantization_steps'],
        wavenet_params['dilations'],
        wavenet_params['filter_width'],
        wavenet_params['residual_channels'],
        wavenet_params['dilation_channels'])

    samples = tf.placeholder(tf.int32)

    next_sample = net.predict_proba(samples)

    saver = tf.train.Saver()
    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    quantization_steps = wavenet_params['quantization_steps']
    waveform = [np.random.randint(quantization_steps)]
    for step in range(args.samples):
        if len(waveform) > args.window:
            window = waveform[-args.window]
        else:
            window = waveform
        prediction = sess.run(
            next_sample,
            feed_dict={samples: window})
        sample = np.random.choice(np.arange(quantization_steps), p=prediction)
        waveform.append(sample)
        print('Sample {:3<d}/{:3<d}: {}'.format(step + 1, args.samples, sample))

    # Undo the companding transformation
    result = net.decode(samples)

    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.train.SummaryWriter(
        os.path.join(logdir, 'generation', datestring))
    tf.audio_summary('generated', result, wavenet_params['sample_rate'])
    summaries = tf.merge_all_summaries()

    summary_out = sess.run(summaries, feed_dict={samples: np.reshape(waveform, [-1, 1])})
    writer.add_summary(summary_out)

    if args.wav_out_path:
        from scipy.io import wavfile

        print('The result saved to {}'.format(args.wav_out_path))
        wavfile.write(args.wav_out_path, wavenet_params['sample_rate'], np.array(waveform).astype(np.uint8))

    print('Finished generating. The result can be viewed in TensorBoard.')

if __name__ == '__main__':
    main()
