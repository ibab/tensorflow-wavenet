from __future__ import division
import librosa
import numpy as np
import tensorflow as tf
import audio_reader
import text_reader
import image_reader
from PIL import Image


def FileReader(data_dir, coord, sample_rate, sample_size,
               silence_threshold, quantization_channels,
               pattern, EPSILON=0.001, raw_type="Audio"):
    if raw_type == "Audio":
        # Allow silence trimming to be skipped by specifying a
        # threshold near zero.
        silence_threshold = silence_threshold if silence_threshold > \
                                                      EPSILON else None
        reader = audio_reader.AudioReader(
                 data_dir, coord,
                 sample_rate=sample_rate,
                 sample_size=sample_size,
                 silence_threshold=silence_threshold,
                 quantization_channels=quantization_channels,
                 pattern=pattern)
    elif raw_type == "Text":
        reader = text_reader.TextReader(data_dir, coord,
                                        sample_size=sample_size,
                                        pattern=pattern)
    elif raw_type == "Image":
        reader = image_reader.ImageReader(data_dir, coord,
                                          sample_size=sample_size,
                                          pattern=pattern)
    return reader


def write_output(waveform, filename, sample_rate, raw_type="Audio"):
    if raw_type == "Image":
        write_img(waveform, filename)
    elif raw_type == "Text":
        write_text(waveform, filename)
    else:
        write_wav(waveform, sample_rate, filename)


def write_img(waveform, filename):
    img = waveform[:-1]
    img = np.array(img)
    img = img.reshape(-1, 1)
    img = img.reshape(64, 64)
    new_img = Image.fromarray(img)
    new_img = new_img.convert('RGB')
    new_img.save(filename)
    print('Updated image file at {}'.format(filename))


def write_text(waveform, filename):
    text = waveform
    y = []
    for index, item in enumerate(text):
        y.append(chr(text[index]))
    print('Prediction is: ', ''.join(str(e) for e in y))
    y = np.array(y)
    np.savetxt(filename, y.reshape(1, y.shape[0]),
               delimiter="", newline="\n", fmt="%s")
    print('Updated text file at {}'.format(filename))


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed_audio(filename,
                      sample_rate,
                      quantization_channels,
                      window_size=8000,
                      silence_threshold=0.1):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)
    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))
    return quantized[:cut_index]


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


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
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.name_scope(name):
        # Pad beforehand to preserve causality.
        filter_width = tf.shape(filter_)[0]
        padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
        padded = tf.pad(value, padding)
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')
        # Remove excess elements at the end.
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, tf.shape(value)[1], -1])
        return result


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = quantization_channels - 1
        # Perform mu-law companding transformation (ITU-T, 1988).
        magnitude = tf.log(1 + mu * tf.abs(audio)) / tf.log(1. + mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        casted = tf.cast(output, tf.float32)
        signal = 2 * (casted / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude
