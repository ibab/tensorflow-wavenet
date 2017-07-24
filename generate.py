from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

# import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from wavenet import WaveNetModel, CsvReader

SAMPLES = 1000
LOGDIR = './output'

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

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
        '--fast_generation',
        type=_str_to_bool,
        default=False,
        help='Use fast generation')
    parser.add_argument(
        '--dat_seed',
        type=str,
        default=None,
        help='The data file to start generation from.. Must include .emo and .pho file.')

    arguments = parser.parse_args()

    return arguments


def write_output(data, filename):
    y = data
    np.savetxt(filename, np.array(y), delimiter=",", newline="\n", fmt="%.10e")
    print('Written CSV file at {}'.format(filename))


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)

    sess = tf.Session()


    # TODO: this does not seem necessary, is it?
    #variables_to_restore = {
    #    var.name[:-2]: var for var in tf.global_variables()
    #    if not ('state_buffer' in var.name or 'pointer' in var.name)}
    #saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver = tf.train.import_meta_graph(args.checkpoint + '.meta', clear_devices=False)
    saver.restore(sess, args.checkpoint)

    # Get the config from the Graph
    config = {}
    for cfg_tensor in tf.get_collection("config"):
        config[cfg_tensor.name.split(":")[0]] = sess.run(cfg_tensor)

    # TODO: should this really include sample_size?
    feed_size = config['receptive_field_size'] + config['sample_size']

    if args.fast_generation:
        next_sample = tf.get_collection("predict_proba_incremental")[0]
    else:
        next_sample = tf.get_collection("predict_proba")[0]

    # TODO: Build up seed placeholders...
    if args.dat_seed:
        raise NotImplementedError("No seed function yet...")

    else:
        data_feed = np.zeros([feed_size,config['data_dim']], dtype=np.float32)
        gc_feed = np.zeros(feed_size, dtype=np.int32)
        lc_feed = np.zeros(feed_size, dtype=np.int32)


    last_sample_timestamp = datetime.now()

    for step in range(args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window_data = data_feed[-1]

        else:
            if len(data_feed) > config['receptive_field_size']:
                window_data = data_feed[-config['receptive_field_size']:]
                window_gc = gc_feed[-config['receptive_field_size']:]
                window_lc = lc_feed[-config['receptive_field_size']:]
            else:
                window_data = data_feed[:]
                window_gc = gc_feed[:]
                window_lc = lc_feed[:]

            outputs = [next_sample]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={'samples:0': window_data, 'gc:0': window_gc, 'lc:0': window_lc})[0]

        data_feed = np.append(data_feed, prediction, axis=0)
        ## TODO: HERE FEED IN THE CONDITIONINGS.
        gc_feed = np.append(gc_feed, 3)
        lc_feed = np.append(lc_feed, 3)

        print(prediction)

        # Show progress only once per second.
        #current_sample_timestamp = datetime.now()
        #time_since_print = current_sample_timestamp - last_sample_timestamp
        #if time_since_print.total_seconds() > 1.:
        #    print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples), end='\r')
        #    last_sample_timestamp = current_sample_timestamp


    # Introduce a newline to clear the carriage return from the progress.
    print()

    # TODO: This could be re-written as an image summary.
    # Save the result as an audio summary.
    # datestring = str(datetime.now()).replace(' ', 'T')
    # writer = tf.train.SummaryWriter(logdir)
    # tf.audio_summary('generated', decode, wavenet_params['sample_rate'])
    # summaries = tf.merge_all_summaries()
    # summary_out = sess.run(summaries,
    #                        feed_dict={samples: np.reshape(waveform, [-1, 1])})
    # writer.add_summary(summary_out)

    # Save the result as a wav file.
    if args.wav_out_path:
        if not args.fast_generation:
            waveform = waveform[net.receptive_field:]
        samp = np.array(waveform).reshape([-1, quantization_channels])
        out = sess.run(decode, feed_dict={samples: samp})
        write_output(out, args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
