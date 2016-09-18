'''Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
'''

import argparse
from datetime import datetime
import glob
import json
import os
import re
import sys
import time

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
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Directory in which to store the logging '
                        'information for TensorBoard.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters.')
    return parser.parse_args()


def create_vctk_inputs(directory, sample_rate=16000):
    '''Loads audio samples and speaker IDs from the VCTK dataset.'''

    # We retrieve each audio sample, the corresponding text, and the speaker id
    audio_glob = directory + '/wav48/*/*.wav'
    audio_filenames = glob.glob(audio_glob)
    if len(audio_filenames) == 0:
        print('No audio files matching {} have been found!'.format(audio_glob))
        sys.exit(1)

    print('Found {} audio files'.format(len(audio_filenames)))
    audio_files = tf.train.string_input_producer(audio_filenames)
    reader = tf.WholeFileReader()
    _, audio_values = reader.read(audio_files)

    # Find the speaker ID and map it into a range [0, ...,  num_speakers].
    dirs = glob.glob(directory + '/wav48/p*')
    speaker_re = re.compile(r'p([0-9]+)')
    ids = [speaker_re.findall(d)[0] for d in dirs]
    speaker_map = {speaker_id: idx for idx, speaker_id in enumerate(ids)}
    speaker = [speaker_map[speaker_re.findall(p)[0]] for p in audio_filenames]

    audio_files = tf.train.string_input_producer(audio_filenames)
    speaker_values = tf.train.input_producer(speaker)

    waveform = ffmpeg.decode_audio(
        audio_values,
        file_format='wav',
        samples_per_second=sample_rate,
        # Corpus uses mono.
        channel_count=1)

    return waveform, speaker_values.dequeue()


def main():
    args = get_arguments()
    datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'train', datestring)

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        audio, speaker = create_vctk_inputs(args.data_dir,
                                            wavenet_params["sample_rate"])

        queue = tf.PaddingFIFOQueue(
            256,  # Queue size.
            ['float32', 'int32'],
            shapes=[(None, 1), ()])
        enqueue = queue.enqueue([audio, speaker])
        # Don't condition on speaker IDs for now.
        audio_batch, _ = queue.dequeue_many(args.batch_size)

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

    init = tf.initialize_all_variables()
    sess.run(init)

    # Load and enqueue examples in the background.
    coord = tf.train.Coordinator()
    qr = tf.train.QueueRunner(queue, [enqueue])
    qr.create_threads(sess, coord=coord, start=True)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver()

    try:
      for step in range(args.num_steps):
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

          if step % 50 == 0 or step == args.num_steps - 1:
              checkpoint_path = os.path.join(logdir, 'model.ckpt')
              print('Storing checkpoint to {}'.format(checkpoint_path))
              saver.save(sess, checkpoint_path, global_step=step)

    finally:
      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
    main()
