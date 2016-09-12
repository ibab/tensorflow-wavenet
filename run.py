
import glob
import re

import threading
import tensorflow as tf
from tensorflow.contrib import ffmpeg

BATCH_SIZE = 3
QUANTIZATION_MU = 255

def create_input_queue(audio_names, txt_names, speakers):
    audio_files = tf.train.string_input_producer(audio_names)
    txt_files = tf.train.string_input_producer(txt_names)

    reader_a = tf.WholeFileReader()
    _, audio_values = reader_a.read(audio_files)
    reader_b = tf.WholeFileReader()
    _, txt_values = reader_a.read(txt_files)

    waveform = ffmpeg.decode_audio(
        audio_values,
        file_format='wav',
        # Downsample to 8khz
        samples_per_second=1<<12,
        # Corpus uses mono
        channel_count=1)

    # Perform mu-law companding transformation (ITU-T, 1988)
    signal = tf.sign(waveform) * (tf.log(1 + QUANTIZATION_MU * tf.abs(waveform))
                                / tf.log(1. + QUANTIZATION_MU))
    quantized = tf.cast((signal + 1) / 2 * QUANTIZATION_MU, tf.int32)

    speaker_values = tf.train.input_producer(speakers)

    return quantized, txt_values, speaker_values.dequeue()


layer_count = 0

def create_causal_dilation(input_batch, dilation=1):
    global layer_count
    layer_count += 1

    wf = tf.Variable(tf.random_normal([1, 2, 256, 256], stddev=0.35, name="filter"))
    wg = tf.Variable(tf.random_normal([1, 2, 256, 256], stddev=0.35, name="gate"))

    # TensorFlow has an operator for convolution with holes
    tmp1 = tf.nn.atrous_conv2d(input_batch, wf,
            rate=dilation,
            padding="SAME",
            name="conv_f")
    tmp2 = tf.nn.atrous_conv2d(input_batch, wg,
            rate=dilation,
            padding="SAME",
            name="conv_g")

    out = tf.tanh(tmp1) * tf.sigmoid(tmp2)

    # Shift output to the right by dilation count so that only current/past
    # values can influence the prediction
    out = tf.slice(out, [0, 0, 0, 0], [-1, -1, tf.shape(out)[2] - dilation, -1])
    out = tf.pad(out, [[0, 0], [0, 0], [dilation, 0], [0, 0]])

    tf.histogram_summary('layer{}_filter'.format(layer_count), wf)
    tf.histogram_summary('layer{}_guard'.format(layer_count), wg)

    w = tf.Variable(tf.random_normal([1, 1, 256, 256], stddev=0.35, name="dense"))
    transformed = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME", name="dense")

    return input_batch + transformed

def create_network(input_batch):

    with tf.name_scope('transform_inputs'):
        waves = tf.reshape(input_batch, [BATCH_SIZE, 1, -1])
        encoded = tf.one_hot(input_batch, depth=QUANTIZATION_MU + 1, dtype=tf.float32)
        encoded = tf.reshape(encoded, [BATCH_SIZE, 1, -1, QUANTIZATION_MU + 1])

    dilations = [1, 2, 4]
    outputs = []
    current_layer = encoded

    for i, dilation in enumerate(dilations):
        with tf.name_scope('layer{}'.format(i)):
            current_layer = create_causal_dilation(current_layer, dilation=dilation)
            outputs.append(current_layer)

    with tf.name_scope('postprocess'):
        # Postprocess sum of outputs using dense layers
        w1 = tf.Variable(tf.random_normal([1, 1, 256, 256], stddev=0.35, name="postprocess1"))
        w2 = tf.Variable(tf.random_normal([1, 1, 256, 256], stddev=0.35, name="postprocess2"))

        summed = tf.nn.relu(tf.add_n(outputs))
        convolved = tf.nn.conv2d(summed, w1, [1, 1, 1, 1], padding="SAME")
        skipped = tf.nn.relu(convolved)
        out = tf.nn.conv2d(skipped, w2, [1, 1, 1, 1], padding="SAME")

    with tf.name_scope('loss'):
        # Shift original input left by one sample, which means that the network
        # will have to predict the immediate future
        shifted = tf.slice(encoded, [0, 0, 1, 0], [-1, -1, tf.shape(encoded)[2] - 1, -1])
        shifted = tf.pad(shifted, [[0, 0], [0, 0], [0, 1], [0, 0]])

        loss = tf.nn.softmax_cross_entropy_with_logits(out, tf.reshape(shifted, [-1, QUANTIZATION_MU + 1]))
        reduced_loss =  tf.reduce_mean(loss)

    tf.scalar_summary('loss', reduced_loss)

    return reduced_loss


def main():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    # We retrieve each audio sample, the corresponding text, and the speaker id
    audio_filenames = glob.glob('./VCTK-Corpus/wav48/**/*.wav', recursive=True)

    # Mysteriously, speaker 315 doesn't have any text files associated with them
    audio_filenames = list(filter(lambda x: not '315' in x, audio_filenames))
    text_filenames = [f.replace('wav48', 'txt').replace('.wav', '.txt')
                        for f in audio_filenames]

    dirs = glob.glob('./VCTK-Corpus/txt/p*')
    SPEAKER_RE = r'p([0-9]+)'
    ids = [re.findall(SPEAKER_RE, d)[0] for d in dirs]
    speaker_map = {speaker_id: idx for idx, speaker_id in enumerate(ids)}
    speaker = [speaker_map[re.findall(SPEAKER_RE, p)[0]] for p in text_filenames]

    with tf.name_scope('create_inputs'):
        audio, txt, speaker = create_input_queue(audio_filenames, text_filenames, speaker)

    queue = tf.PaddingFIFOQueue(1000, ["int32", "string", "int32"], shapes=[(None, 1), (), ()])
    enqueue = queue.enqueue([audio, txt, speaker])
    audio_op, txt_op, speaker_op = queue.dequeue_many(BATCH_SIZE)

    loss = create_network(audio_op)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.10)

    trainable = tf.trainable_variables()

    optim = optimizer.minimize(loss, var_list=trainable)

    writer = tf.train.SummaryWriter('./log')
    writer.add_graph(tf.get_default_graph())
    summaries = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    sess.run(init)

    coord = tf.train.Coordinator()

    # Enqueue examples in the background
    qr = tf.train.QueueRunner(queue, [enqueue])
    qr.create_threads(sess, coord=coord, start=True)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1000):
        summary, loss_ = sess.run([summaries, loss])
        sess.run(optim)
        writer.add_summary(summary, i)
        print('Loss: {}'.format(loss_))

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
