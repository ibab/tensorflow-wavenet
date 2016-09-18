
import tensorflow as tf

def Print(x):
    return tf.Print(x, [tf.shape(x)], summarize=30)

def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[2] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, 1, dilation, shape[3]])
        transposed = tf.transpose(reshaped, perm=[2, 1, 0, 3])
        return tf.reshape(transposed, [shape[0] * dilation, 1, -1, shape[3]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, 1, -1, shape[3]])
        transposed = tf.transpose(prepared, perm=[2, 1, 0, 3])
        return tf.reshape(transposed, [tf.div(shape[0], dilation), 1, -1, shape[3]])


def causal_conv(value, filter_, dilation, name=None):
    with tf.name_scope('causal_conv'):
        # Pad beforehand to preserve causality
        filter_width = tf.shape(filter_)[1]
        padded = tf.pad(value, [[0, 0], [0, 0], [(filter_width - 1) * dilation, 0], [0, 0]])
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv2d(transformed, filter_, strides=[1, 1, 1, 1], padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv2d(padded, filter_, strides=[1, 1, 1, 1], padding='SAME')
        # Remove excess elements at the end
        result = tf.slice(restored,
                         [0, 0, 0, 0],
                         [-1, -1, tf.shape(value)[2], -1])
        return result
