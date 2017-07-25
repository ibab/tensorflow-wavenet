from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
from termcolor import colored
import os

# import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

import json

from socket import *
import thread

"""
Training script for the EveNet network.

Telnet:

PHO AX
EMO Neutral

see reader config for valid...

"""


HOST = '127.0.0.1'# must be input parameter @TODO
PORT = 9999 # must be input parameter @TODO

SAMPLES = 1000
LOGDIR = './output'

CURRENT_EMOTION = "Neutral"
CURRENT_PHONEME = "SIL"

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
        '--out_path',
        type=str,
        default=None,
        help='Directory in which to store the output')
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
    parser.add_argument(
        '--emotion',
        type=str,
        default=CURRENT_EMOTION,
        help='The emotion to be synthesized.')
    parser.add_argument(
        '--phoneme',
        type=str,
        default=CURRENT_PHONEME,
        help='The phoneme to be synthesized')
    parser.add_argument(
        '--reader_config',
        type=str,
        default="./reader_config.json",
        help='The configuration file for mapping phonemes and emotions to GC and LC')

    arguments = parser.parse_args()

    return arguments


def write_output(data, filename):
    y = data
    np.savetxt(filename, np.array(y), delimiter=",", newline="\n", fmt="%.10e")
    print('Written CSV file at {}'.format(filename))


with open(get_arguments().reader_config) as json_file:
    mapping_config = json.load(json_file)

def get_phoneme_id(phoneme):
    return mapping_config['phoneme_categories'].index(phoneme)

def get_emotion_id(emotion):
    return mapping_config['emotion_categories'].index(emotion)


""" Network stuff """
def listener(serversock, arg):
    while 1:
        clientsock, addr = serversock.accept()
        thread.start_new_thread(handler, (clientsock, addr))

def handler(clientsock,addr):
    global CURRENT_EMOTION, CURRENT_PHONEME
    while 1:
        data = clientsock.recv(1024)
        command = data.split(" ")
        if len(command)==2:
            arg = command[1].strip()

            if command[0]=="EMO":
                if arg in mapping_config['emotion_categories']:
                    CURRENT_EMOTION = arg
            if command[0]=="PHO":
                if arg in mapping_config['phoneme_categories']:
                    CURRENT_PHONEME = arg

def start_socket():
    print(colored("Listening on port %i" % PORT, 'cyan'))
    serversock = socket(AF_INET, SOCK_STREAM)
    serversock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serversock.bind((HOST, PORT))
    serversock.listen(5)

    thread.start_new_thread(listener, (serversock, 0))







def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)

    start_socket()

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

    if args.fast_generation:
        next_sample = tf.get_collection("predict_proba_incremental")[0]
    else:
        next_sample = tf.get_collection("predict_proba")[0]

    # TODO: Build up seed placeholders...
    if args.dat_seed:
        raise NotImplementedError("No seed function yet...")
    else:
        data_feed = np.zeros([config['receptive_field_size'],config['data_dim']], dtype=np.float32)
        gc_feed = np.zeros(config['receptive_field_size'], dtype=np.int32)
        lc_feed = np.zeros(config['receptive_field_size'], dtype=np.int32)


    last_sample_timestamp = datetime.now()

    try:
        for step in range(args.samples):
            if args.fast_generation:
                outputs = [next_sample]
                outputs.extend(net.push_ops)
                window_data = data_feed[-1]

                # See Alex repository for feeding in initial samples...
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
            gc_feed = np.append(gc_feed, get_emotion_id(CURRENT_EMOTION))
            lc_feed = np.append(lc_feed, get_phoneme_id(CURRENT_PHONEME))

            # TODO: Output to ROS here...
            print("%5i %s %s \n %s" % (step,
                                      colored(CURRENT_EMOTION, 'blue'),
                                      colored(CURRENT_PHONEME, 'white', 'on_grey', attrs=['bold']),
                                      colored(str(prediction), 'grey')))
    except KeyboardInterrupt:
        pass

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
    if args.out_path:
        if not args.fast_generation:
            data_feed = data_feed[config['receptive_field_size']:]
        samp = np.array(data_feed).reshape([-1, config['data_dim']])
        write_output(samp, args.out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
