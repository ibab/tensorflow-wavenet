'''Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
'''

from __future__ import print_function

import argparse
from datetime import datetime
import glob
import json
import os
import re
import sys
import time
import threading
import numpy as np
import fnmatch
import librosa

import tensorflow as tf
from tensorflow.contrib import ffmpeg
import tensorflow.python.client.timeline as timeline

from wavenet import WaveNet

BATCH_SIZE = 2
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR = './logdir'
NUM_STEPS = 4000
LEARNING_RATE = 0.02
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir), end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("")
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)

    return logdir


def scan_directory(directory):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.wav'):
            files.append(os.path.join(root, filename))
    return files


def iterate_through_directory(directory, sample_rate):
    files = scan_directory(directory)
    for f in files:
        audio, sr = librosa.load(f, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, 0


def iterate_through_vctk(directory, sample_rate):
    files = scan_directory(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for j,f in enumerate(files):
        audio, sr = librosa.load(f, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        speaker_id, recording_id = [int(i) for i in speaker_re.findall(f)[0]]
        yield audio, speaker_id


def validate_directories(args):
    """
    Validate and arrange directory related arguments.
    """

    # Validation
    # ----------

    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError("--logdir and --restore_from cannot be "
                         "specified at the same time. This is to keep "
                         "your previous model from unexpected overwrites.\n"
                         "Use --logdir_root to specify the root of the directory "
                         "which will be automatically created with current date "
                         "and time, or use only --logdir to just continue the "
                         "training from the last checkpoint.")

    # Arrangement
    # -----------

    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print ('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from is exclusive,
        # So it is guranteed the logdir here is newly created.

        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


class CustomRunner(object):
    def __init__(self, args, wavenet_params, coord):
        self.args = args
        self.wavenet_params = wavenet_params
        self.coord = coord
        self.threads = []
        self.dataX = tf.placeholder(dtype=tf.float32, shape=None)
        self.dataY = tf.placeholder(dtype=tf.int32)
        self.queue = tf.PaddingFIFOQueue(
            256,  # Queue size.
            ['float32', 'int32'],
            shapes=[(None, 1), ()])
        self.enqueue = self.queue.enqueue([self.dataX, self.dataY])

    def get_inputs(self):
        self.audio_batch, _ = self.queue.dequeue_many(self.args.batch_size)
        return self.audio_batch, _

    def thread_main(self, sess):
        for audio, label in iterate_through_vctk(self.args.data_dir, self.wavenet_params["sample_rate"]):
            if self.coord.should_stop():
                self.stop_threads()
            sess.run(self.enqueue, feed_dict={self.dataX:audio, self.dataY:label})

    def stop_threads():
        for t in self.threads:
            t.stop()

    def start_threads(self, sess, n_threads=1):
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            self.threads.append(t)
        return self.threads


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    logdir_root = directories['logdir_root']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into arbitrary location.
    is_new_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # create coordinator
    coord = tf.train.Coordinator()
    
    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        custom_runner = CustomRunner(args, wavenet_params, coord)
        audio_batch, _ = custom_runner.get_inputs()
    
    # Create network.
    net = WaveNet(args.batch_size,
                  wavenet_params["quantization_steps"],
                  wavenet_params["dilations"],
                  wavenet_params["filter_width"],
                  wavenet_params["residual_channels"],
                  wavenet_params["dilation_channels"])
    loss = net.loss(audio_batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard.
    writer = tf.train.SummaryWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.merge_all_summaries()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.initialize_all_variables()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver()

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_new_training or saved_global_step is None:
            # For "new" training with using pre-trained model,
            # We should ignore saved_global_step

            # The training step is start from saved_global_step + 1
            # Therefore put -1 here if the new training starts.
            saved_global_step = -1

    except:
        print("Something is wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    custom_runner.start_threads(sess)

    try:
        for step in range(saved_global_step + 1, args.num_steps):
            start_time = time.time()
            if args.store_metadata and step % 50 == 0:
                # Slow run that stores extra information for debugging.
                print('Storing metadata')
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                summary, loss_value, _ = sess.run(
                    [summaries, loss, optim],
                    options=run_options,
                    run_metadata=run_metadata)
                writer.add_summary(summary, step)
                writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step))
                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                summary, loss_value, _ = sess.run([summaries, loss, optim])
                writer.add_summary(summary, step)

            duration = time.time() - start_time
            print('step %d - loss = %.3f, (%.3f sec/step)' % (step, loss_value, duration))

            if step % 50 == 0:
                save(saver, sess, logdir, step)

    finally:
        coord.request_stop()
        coord.join(threads)
        

if __name__ == '__main__':
    main()
