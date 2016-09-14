
import glob
import re
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.contrib import ffmpeg
import tensorflow.python.client.timeline as timeline

from wavenet import WaveNet

BATCH_SIZE = 1
CHANNELS = 256
DATA_DIRECTORY = './VCTK-Corpus'
FILTER_WIDTH = 2
LOGDIR = './logdir'

def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once')
    parser.add_argument('--channels', type=int, default=CHANNELS,
                        help='Number of possible waveform amplitude values')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with TensorBoard')
    parser.add_argument('--filter_width', type=int, default=FILTER_WIDTH,
                        help='Width of the filters to use in the causal dilated convolutions')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Directory in which to store the logging information for TensorBoard')
    return parser.parse_args()


def create_vctk_inputs(directory):
    '''Create an input pipeline that loads audio samples and speaker IDs from
    the VCTK dataset.
    '''

    # We retrieve each audio sample, the corresponding text, and the speaker id
    audio_filenames = glob.glob(directory + '/wav48/*/*.wav')
    audio_files = tf.train.string_input_producer(audio_filenames)
    reader = tf.WholeFileReader()
    _, audio_values = reader.read(audio_files)

    # Find the speaker ID and map it into a range [0, ...,  num_speakers]
    dirs = glob.glob(directory + '/wav48/p*')
    speaker_re = r'p([0-9]+)'
    ids = [re.findall(speaker_re, d)[0] for d in dirs]
    speaker_map = {speaker_id: idx for idx, speaker_id in enumerate(ids)}
    speaker = [speaker_map[re.findall(speaker_re, p)[0]] for p in audio_filenames]

    audio_files = tf.train.string_input_producer(audio_filenames)
    speaker_values = tf.train.input_producer(speaker)

    waveform = ffmpeg.decode_audio(
        audio_values,
        file_format='wav',
        # Downsample to 16khz
        samples_per_second=1<<13,
        # Corpus uses mono
        channel_count=1)

    return waveform, speaker_values.dequeue()


def main():
    args = get_arguments()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    # Load raw waveform from VCTK corpus
    with tf.name_scope('create_inputs'):
        audio, speaker = create_vctk_inputs(args.data_dir)

        queue = tf.PaddingFIFOQueue(
            256,
            ['float32', 'int32'],
            shapes=[(None, 1), ()])
        enqueue = queue.enqueue([audio, speaker])
        # Don't condition on speaker IDs for now
        audio_batch, _ = queue.dequeue_many(args.batch_size)

    # Create network
    dilations = [1, 2, 4, 8, 16]
    net = WaveNet(args.batch_size,
                  args.channels,
                  dilations,
                  filter_width=args.filter_width)
    loss = net.loss(audio_batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.10)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard
    writer = tf.train.SummaryWriter('./logdir/TRAIN-{}'.format(str(datetime.now())))
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    sess.run(init)

    # Load and enqueue examples in the background
    coord = tf.train.Coordinator()
    qr = tf.train.QueueRunner(queue, [enqueue])
    qr.create_threads(sess, coord=coord, start=True)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1000):
        if args.store_metadata and i % 50 == 0:
            # Slow run that stores extra information for debugging
            print('Storing metadata')
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            summary, loss_value, _ = sess.run(
                [summaries, loss, optim],
                options=run_options,
                run_metadata=run_metadata)
            writer.add_summary(summary, i)
            writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(i))
            tl = timeline.Timeline(run_metadata.step_stats)
            with open('./timeline.trace', 'w') as f:
                f.write(tl.generate_chrome_trace_format(show_memory=True))

        else:
            summary, loss_value, _ = sess.run([summaries, loss, optim])
            writer.add_summary(summary, i)

        print('Loss: {}'.format(loss_value))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
