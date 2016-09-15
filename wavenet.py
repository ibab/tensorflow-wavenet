import tensorflow as tf

class WaveNet(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        channels = 2**8  # Quantize to 256 possible amplitude values.
        net = WaveNet(batch_size, channels, dilations)
        loss = net.loss(input_batch)
    '''

    def __init__(self, batch_size, channels, dilations, filter_width=2):
        self.batch_size = batch_size
        self.channels = channels
        self.dilations = dilations
        self.filter_width = filter_width

    def _create_dilation_layer(self, input_batch, layer_index, dilation):
        '''Adds a single causal dilated convolution layer.'''

        # The filter widths can be configured as a hyperparameter.
        # It corresponds to the number of samples we combine at once
        weights_filter = tf.Variable(tf.truncated_normal(
            [1, self.filter_width, self.channels, self.channels],
            stddev=0.2,
            name="filter"))
        weights_gate = tf.Variable(tf.truncated_normal(
            [1, self.filter_width, self.channels, self.channels],
            stddev=0.2, name="gate"))

        # TensorFlow has an operator for convolution with holes.
        conv_filter = tf.nn.atrous_conv2d(input_batch, weights_filter,
                                          rate=dilation,
                                          padding="VALID",
                                          name="conv_filter")
        conv_gate = tf.nn.atrous_conv2d(input_batch, weights_gate,
                                        rate=dilation,
                                        padding="VALID",
                                        name="conv_gate")

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # Pad output with zeros from the left to ensure that a given pixel only
        # uses current/past values
        out = tf.pad(out, [[0, 0], [0, 0], [dilation, 0], [0, 0]])

        weights_dense = tf.Variable(tf.truncated_normal(
            [1, 1, self.channels, self.channels], stddev=0.2, name="dense"))
        transformed = tf.nn.conv2d(out, weights_dense, strides=[1] * 4,
                                   padding="SAME", name="dense")

        layer = 'layer{}'.format(layer_index)
        tf.histogram_summary(layer + '_filter', weights_filter)
        tf.histogram_summary(layer + '_gate', weights_gate)
        tf.histogram_summary(layer + '_dense', weights_dense)

        return transformed, input_batch + transformed

    def _preprocess(self, audio):
        '''Quantizes waveform amplitudes.'''
        with tf.name_scope('preprocessing'):
            mu = self.channels - 1
            # Perform mu-law companding transformation (ITU-T, 1988).
            magnitude = tf.log(1 + mu * tf.abs(audio)) / tf.log(1. + mu)
            signal = tf.sign(audio) * magnitude
            # Quantize signal to the specified number of levels
            quantized = tf.cast((signal + 1) / 2 * mu, tf.int32)

        return quantized


    def decode(self, output):
        mu = self.channels
        output = tf.cast(output, tf.float32)
        y = (2 * output  - 1) / mu
        x = tf.sign(y) * (tf.exp(y * tf.log(1. + mu)) - 1) / mu

        return x


    def _create_network(self, input_batch):
        outputs = []
        current_layer = input_batch

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer,
                        layer_index,
                        dilation=dilation)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = tf.Variable(tf.truncated_normal(
                [1, 1, self.channels, self.channels], stddev=0.3,
                name="postprocess1"))
            w2 = tf.Variable(tf.truncated_normal(
                [1, 1, self.channels, self.channels], stddev=0.3,
                name="postprocess2"))

            tf.histogram_summary('postprocess1_weights', w1)
            tf.histogram_summary('postprocess2_weights', w2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv2d(transformed1, w1, [1] * 4, padding="SAME")
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv2d(transformed2, w2, [1] * 4, padding="SAME")

        return conv2


    def _one_hot(self, input_batch):
        # One-hot encode waveform amplitudes, so we can define the network
        # as a categorical distribution over possible amplitudes.
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(input_batch, depth=self.channels,
                                 dtype=tf.float32)
            encoded = tf.reshape(encoded,
                                 [self.batch_size, 1, -1, self.channels])

        return encoded


    def predict_proba(self, waveform, name='wavenet'):
        with tf.variable_scope(name):
            encoded = self._one_hot(waveform)
            raw_output = self._create_network(encoded)
            out = tf.reshape(raw_output, [-1, self.channels])
            proba = tf.nn.softmax(out)
            last = tf.slice(proba, [tf.shape(proba)[0] - 1, 0], [1, self.channels])
            return tf.reshape(last, [-1])


    def loss(self, input_batch, name='wavenet'):
        with tf.variable_scope(name):
            input_batch = self._preprocess(input_batch)
            encoded = self._one_hot(input_batch)
            raw_output = self._create_network(encoded)

            with tf.name_scope('loss'):
                # Shift original input left by one sample, which means that
                # each output sample has to predict the next input sample.
                shifted = tf.slice(encoded, [0, 0, 1, 0],
                                   [-1, -1, tf.shape(encoded)[2] - 1, -1])
                shifted = tf.pad(shifted, [[0, 0], [0, 0], [0, 1], [0, 0]])

                prediction = tf.reshape(raw_output, [-1, self.channels])
                loss = tf.nn.softmax_cross_entropy_with_logits(
                    prediction,
                    tf.reshape(shifted, [-1, self.channels]))
                reduced_loss = tf.reduce_mean(loss)

                tf.scalar_summary('loss', reduced_loss)

        return reduced_loss
