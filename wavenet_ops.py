
import tensorflow as tf


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name=None):
    with tf.name_scope('causal_conv'):
        # Pad beforehand to preserve causality
        filter_width = tf.shape(filter_)[0]
        padded = tf.pad(value, [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]])
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')
        # Remove excess elements at the end
        result = tf.slice(restored,
                         [0, 0, 0],
                         [-1, tf.shape(value)[1], -1])
        return result
