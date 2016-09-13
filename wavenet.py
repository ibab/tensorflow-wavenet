import tensorflow as tf

class WaveNet(object):

    def __init__(self, batch_size, channels, dilations):
        self.batch_size = batch_size
        self.channels = channels
        self.dilations = dilations

    def _create_dilation_layer(self, input_batch, i, dilation):
        '''Adds a single causal dilated convolution layer'''

        # TODO Which filter width should be used?
        # The pictures in the paper seem to suggest 2, but larger filters would
        # also make sense and could be more performant
        wf = tf.Variable(tf.truncated_normal([1, 2, 256, 256], stddev=0.2, name="filter"))
        wg = tf.Variable(tf.truncated_normal([1, 2, 256, 256], stddev=0.2, name="gate"))

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

        w = tf.Variable(tf.truncated_normal([1, 1, 256, 256], stddev=0.20, name="dense"))
        transformed = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME", name="dense")

        tf.histogram_summary('layer{}_filter'.format(i), wf)
        tf.histogram_summary('layer{}_guard'.format(i), wg)
        tf.histogram_summary('layer{}_weights'.format(i), w)

        return input_batch + transformed

    def _preprocess(self, audio):
        '''Quantize waveform amplitudes
        '''
        mu = self.channels - 1
        # Perform mu-law companding transformation (ITU-T, 1988)
        signal = tf.sign(audio) * (tf.log(1 + mu * tf.abs(audio))
                                    / tf.log(1. + mu))
        quantized = tf.cast((signal + 1) / 2 * mu, tf.int32)

        return quantized

    def _create_network(self, input_batch):
        outputs = []
        current_layer = input_batch

        # Add all defined dilation layers
        for i, dilation in enumerate(self.dilations):
            with tf.name_scope('layer{}'.format(i)):
                current_layer = self._create_dilation_layer(
                        current_layer,
                        i,
                        dilation=dilation)
                outputs.append(current_layer)

        with tf.name_scope('postprocess'):
            # Some 1x1 convolution + ReLU layers to postprocess the output
            w1 = tf.Variable(tf.truncated_normal([1, 1, 256, 256], stddev=0.3, name="postprocess1"))
            w2 = tf.Variable(tf.truncated_normal([1, 1, 256, 256], stddev=0.3, name="postprocess2"))

            tf.histogram_summary('postprocess1_weights', w1)
            tf.histogram_summary('postprocess2_weights', w2)

            # We skip connections from the outputs of each layers, adding them all up here
            summed = tf.add_n(outputs)
            transformed1 = tf.nn.relu(summed)
            conv1 = tf.nn.conv2d(transformed1, w1, [1, 1, 1, 1], padding="SAME")
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv2d(transformed2, w2, [1, 1, 1, 1], padding="SAME")

        return conv2

    def loss(self, input_batch):
        input_batch = self._preprocess(input_batch)

        # One-hot encode waveform amplitudes, so we can define the network as a
        # categorical distribution over possible amplitudes
        with tf.name_scope('one_hot_encode'):
            waves = tf.reshape(input_batch, [self.batch_size, 1, -1])
            encoded = tf.one_hot(input_batch, depth=self.channels, dtype=tf.float32)
            encoded = tf.reshape(encoded, [self.batch_size, 1, -1, self.channels])

        raw_output = self._create_network(encoded)

        with tf.name_scope('loss'):
            # Shift original input left by one sample, which means that each
            # output pixel has to predict the next input pixel
            shifted = tf.slice(encoded, [0, 0, 1, 0], [-1, -1, tf.shape(encoded)[2] - 1, -1])
            shifted = tf.pad(shifted, [[0, 0], [0, 0], [0, 1], [0, 0]])

            loss = tf.nn.softmax_cross_entropy_with_logits(raw_output, tf.reshape(shifted, [-1, self.channels]))
            reduced_loss =  tf.reduce_mean(loss)

            tf.scalar_summary('loss', reduced_loss)

        return reduced_loss
