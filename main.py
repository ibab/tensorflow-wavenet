
import glob
import re

import threading
import tensorflow as tf
from tensorflow.contrib import ffmpeg

from wavenet import WaveNet

BATCH_SIZE = 1
CHANNELS = 256
DATA_DIRECTORY='./VCTK-Corpus'

def create_vctk_inputs(directory):
    # TODO make sure that text is matched correctly to the samples

    # We retrieve each audio sample, the corresponding text, and the speaker id
    audio_filenames = glob.glob(directory + '/wav48/*/*.wav')

    # Mysteriously, speaker 315 doesn't have any text files associated with them
    audio_filenames = list(filter(lambda x: not '315' in x, audio_filenames))
    text_filenames = [f.replace('wav48', 'txt').replace('.wav', '.txt')
                        for f in audio_filenames]

    # Find the speaker ID and map it into a range [0, ...,  num_speakers]
    dirs = glob.glob(directory + '/txt/p*')
    SPEAKER_RE = r'p([0-9]+)'
    ids = [re.findall(SPEAKER_RE, d)[0] for d in dirs]
    speaker_map = {speaker_id: idx for idx, speaker_id in enumerate(ids)}
    speaker = [speaker_map[re.findall(SPEAKER_RE, p)[0]] for p in text_filenames]

    audio_files = tf.train.string_input_producer(audio_filenames)
    txt_files = tf.train.string_input_producer(text_filenames)
    speaker_values = tf.train.input_producer(speaker)

    reader_a = tf.WholeFileReader()
    _, audio_values = reader_a.read(audio_files)
    reader_b = tf.WholeFileReader()
    _, txt_values = reader_a.read(txt_files)

    waveform = ffmpeg.decode_audio(
        audio_values,
        file_format='wav',
        # Downsample to 16khz
        samples_per_second=1<<13,
        # Corpus uses mono
        channel_count=1)

    return waveform, txt_values, speaker_values.dequeue()

def main():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    # Load raw waveform from VCTK corpus
    with tf.name_scope('create_inputs'):
        audio, txt, speaker = create_vctk_inputs(DATA_DIRECTORY)

    queue = tf.PaddingFIFOQueue(
        256,
        ["float32", "string", "int32"],
        shapes=[(None, 1), (), ()])
    enqueue = queue.enqueue([audio, txt, speaker])
    audio_batch, _, _ = queue.dequeue_many(BATCH_SIZE)

    # Create network
    dilations = [1, 2, 4, 8, 16, 32]
    net = WaveNet(BATCH_SIZE, CHANNELS, dilations)
    loss = net.loss(audio_batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.10)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard
    writer = tf.train.SummaryWriter('./logdir')
    writer.add_graph(tf.get_default_graph())
    summaries = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    sess.run(init)

    # Enqueue examples in the background
    coord = tf.train.Coordinator()
    qr = tf.train.QueueRunner(queue, [enqueue])
    qr.create_threads(sess, coord=coord, start=True)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1000):
        summary, loss_value, _ = sess.run([summaries, loss, optim])
        writer.add_summary(summary, i)
        print('Loss: {}'.format(loss_value))

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
