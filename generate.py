
import argparse
from datetime import datetime
import json
import os

import numpy as np
import tensorflow as tf

from wavenet import WaveNet

SAMPLES = 100
LOGDIR = './logdir'
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
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters')
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
        wavenet_params['filter_width'])

    samples = tf.placeholder(tf.int32)

    next_sample = net.predict_proba(samples)

    saver = tf.train.Saver()
    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)
    print('Finished')

    waveform = [0]

    quantization_steps = wavenet_params['quantization_steps']
    for step in range(args.samples):
        prediction = sess.run(next_sample, feed_dict={samples: waveform})
        sample = np.random.choice(np.arange(quantization_steps), p=prediction)
        waveform.append(sample)
        print('Sample {:3<d}/{:3<d}: {}'.format(step + 1, args.samples, sample))

    # Undo the companding transformation
    result = net.decode(samples)

    writer = tf.train.SummaryWriter(
        os.path.join(logdir, 'generation', str(datetime.now())))
    tf.audio_summary('generated', result, wavenet_params['sample_rate'])
    summaries = tf.merge_all_summaries()

    summary_out = sess.run(summaries, feed_dict={samples: np.reshape(waveform, [-1, 1])})
    writer.add_summary(summary_out)

    print('Finished generating. The result can be viewed in TensorBoard.')

if __name__ == '__main__':
    main()
