import argparse
from datetime import datetime
import json
import os
import librosa

import numpy as np
import tensorflow as tf

from wavenet import WaveNet

SAMPLES = 16000
LOGDIR = './logdir'
WINDOW = 8000
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None

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
    parser.add_argument('--save_every', type=int, default=SAVE_EVERY,
                        help='How many samples before saving in-progress wav')
    parser.add_argument('--wav_seed', type=str, default=None,
                        help='The wav file to start generation from')
    return parser.parse_args()

def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

def create_seed(filename, sample_rate, quantization_steps, percent=50):
    audio = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio[0]

    mu = quantization_steps - 1
    magnitude = np.log(1 + mu * np.abs(audio)) / np.log(1. + mu)
    signal = np.sign(audio) * magnitude
    quantized = (signal + 1) / 2 * mu
    quantized = quantized.astype(np.int32)

    cut_index = int(len(quantized) * (percent / 100))
    return quantized[:cut_index].tolist()

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
        wavenet_params['dilation_channels'],
        wavenet_params['use_biases'])

    samples = tf.placeholder(tf.int32)

    next_sample = net.predict_proba(samples)

    saver = tf.train.Saver()
    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    decode = net.decode(samples)

    quantization_steps = wavenet_params['quantization_steps']
    if args.wav_seed:
        waveform = create_seed(args.wav_seed, wavenet_params['sample_rate'], quantization_steps)
    else:
        waveform = np.random.randint(quantization_steps, size=(1,)).tolist()
    for step in range(args.samples):
        if len(waveform) > args.window:
            window = waveform[-args.window:]
        else:
            window = waveform
        prediction = sess.run(
            next_sample,
            feed_dict={samples: window})
        sample = np.random.choice(np.arange(quantization_steps), p=prediction)
        waveform.append(sample)
        print('Sample {:3<d}/{:3<d}: {}'.format(step + 1, args.samples, sample))
        if (args.wav_out_path
            and args.save_every
            and (step + 1) % args.save_every == 0):

            out = sess.run(decode, feed_dict={samples: waveform})
            write_wav(out,
                      wavenet_params['sample_rate'],
                      args.wav_out_path)

    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.train.SummaryWriter(
        os.path.join(logdir, 'generation', datestring))
    tf.audio_summary('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.merge_all_summaries()

    summary_out = sess.run(summaries, feed_dict={samples: np.reshape(waveform, [-1, 1])})
    writer.add_summary(summary_out)

    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        write_wav(out,
                  wavenet_params['sample_rate'],
                  args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')

if __name__ == '__main__':
    main()
