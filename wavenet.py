import tensorflow as tf

from wavenet_ops import causal_conv, mu_law_encode


class WaveNet(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        net = WaveNet(batch_size, dilations, filter_width,
                      residual_channels, dilation_channel)
        loss = net.loss(input_batch)
    '''
    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 use_biases=False,
                 fast_generation=True):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.fast_generation = fast_generation
        if self.fast_generation and self.use_biases:
            raise NotImplementedError('Biases not implemented for fast generation.')

    # A single causal convolution layer that can change the number of channels.
    def _create_causal_layer(self, input_batch, in_channels, out_channels):
        '''Creates a single causal convolution layer.'''
        with tf.name_scope('causal_layer'):
            weights_filter = tf.Variable(tf.truncated_normal(
                [self.filter_width, in_channels, out_channels],
                stddev=0.2,
                name="filter"))
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self,
                               input_batch,
                               layer_index,
                               dilation,
                               in_channels,
                               dilation_channels,
                               skip_channels):
        '''Creates a single causal dilated convolution layer.'''
        weights_filter = tf.Variable(tf.truncated_normal(
            [self.filter_width, in_channels, dilation_channels],
            stddev=0.2, name="filter"))
        weights_gate = tf.Variable(tf.truncated_normal(
            [self.filter_width, in_channels, dilation_channels],
            stddev=0.2, name="gate"))

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if self.use_biases:
            biases_filter = tf.Variable(
                tf.constant(0.0, shape=[dilation_channels]),
                name="filter_biases")
            biases_gate = tf.Variable(
                tf.constant(0.0, shape=[dilation_channels]),
                name="gate_biases")
            conv_filter = tf.add(conv_filter, biases_filter)
            conv_gate = tf.add(conv_gate, biases_gate)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        weights_dense = tf.Variable(tf.truncated_normal(
            [1, dilation_channels, in_channels], stddev=0.2, name="dense"))
        transformed = tf.nn.conv1d(out, weights_dense, stride=1,
                                   padding="SAME", name="dense")

        # The 1x1 conv to produce the skip contribution.
        weights_skip = tf.Variable(tf.truncated_normal(
            [1, dilation_channels, skip_channels], stddev=0.01),
                                   name="skip")
        skip_contribution = tf.nn.conv1d(out, weights_skip, stride=1,
                                         padding="SAME", name="skip")

        if self.use_biases:
            biases_dense = tf.Variable(tf.constant(0.0, shape=[in_channels]),
                                       name="dense_biases")
            transformed = tf.add(transformed, biases_dense)
            biases_skip = tf.Variable(tf.constant(0.0, shape=[skip_channels]),
                                      name="skip_biases")
            skip_contribution = tf.add(skip_contribution, biases_skip)

        layer = 'layer{}'.format(layer_index)
        tf.histogram_summary(layer + '_filter', weights_filter)
        tf.histogram_summary(layer + '_gate', weights_gate)
        tf.histogram_summary(layer + '_dense', weights_dense)
        tf.histogram_summary(layer + '_skip', weights_skip)
        if self.use_biases:
            tf.histogram_summary(layer + '_biases_filter', biases_filter)
            tf.histogram_summary(layer + '_biases_gate', biases_gate)
            tf.histogram_summary(layer + '_biases_dense', biases_dense)
            tf.histogram_summary(layer + '_biases_skip', biases_skip)

        return skip_contribution, input_batch + transformed

    def _apply_weights(self, input_batch, state_batch, weights):
        weights_recurrent = weights[0, :, :]
        weights_embedding = weights[1, :, :]

        output = tf.matmul(input_batch, weights_embedding) + tf.matmul(state_batch,
                                                                       weights_recurrent)
        return output

    def _generator_causal_layer(self, input_batch, state_batch, in_channels, out_channels):
        with tf.name_scope('causal_layer'):
            weights_filter = tf.Variable(tf.truncated_normal(
                [self.filter_width, in_channels, out_channels],
                stddev=0.2,
                name="filter"))

            output = self._apply_weights(input_batch, state_batch, weights_filter)            
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index, dilation,
                                  in_channels, dilation_channels):
        weights_filter = tf.Variable(tf.truncated_normal(
            [self.filter_width, in_channels, dilation_channels],
            stddev=0.2,
            name="filter"))
        weights_gate = tf.Variable(tf.truncated_normal(
            [self.filter_width, in_channels, dilation_channels],
            stddev=0.2, name="gate"))

        output_filter = self._apply_weights(input_batch, state_batch, weights_filter)
        output_gate = self._apply_weights(input_batch, state_batch, weights_gate)

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = tf.Variable(tf.truncated_normal(
            [1, dilation_channels, in_channels], stddev=0.2, name="dense"))
        transformed = tf.matmul(out, weights_dense[0, :, :])
        layer = 'layer{}'.format(layer_index)

        return transformed, input_batch + transformed

    def _create_network(self, input_batch):
        '''Creates a WaveNet network.'''
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        current_layer = self._create_causal_layer(current_layer,
                                                  self.quantization_channels,
                                                  self.residual_channels)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer,
                        layer_index,
                        dilation,
                        self.residual_channels,
                        self.dilation_channels,
                        self.skip_channels)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = tf.Variable(tf.truncated_normal(
                [1, self.skip_channels, self.skip_channels],
                stddev=0.3, name="postprocess1"))
            w2 = tf.Variable(tf.truncated_normal(
                [1, self.skip_channels, self.quantization_channels],
                stddev=0.3, name="postprocess2"))
            if self.use_biases:
                b1 = tf.Variable(
                    tf.constant(0.0, shape=[self.skip_channels]),
                    name="postprocess1_bias")
                b2 = tf.Variable(
                    tf.constant(0.0, shape=[self.quantization_channels]),
                    name="postprocess2_bias")

            tf.histogram_summary('postprocess1_weights', w1)
            tf.histogram_summary('postprocess2_weights', w2)
            if self.use_biases:
                tf.histogram_summary('postprocess1_biases', b1)
                tf.histogram_summary('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

        return conv2
    
    def _create_generator(self, input_batch):
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch

        q = tf.FIFOQueue(1, dtypes=tf.float32, shapes=(self.batch_size, self.quantization_channels))
        init = q.enqueue_many(tf.zeros((1, self.batch_size, self.quantization_channels)))
    
        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)
        
        current_layer = self._generator_causal_layer(current_layer,
                                                     current_state,
                                                     self.quantization_channels,
                                                     self.residual_channels)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    
                    q = tf.FIFOQueue(dilation, dtypes=tf.float32, shapes=(self.batch_size, self.residual_channels))
                    init = q.enqueue_many(tf.zeros((dilation, self.batch_size, self.residual_channels)))
    
                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    
                    output, current_layer = self._generator_dilation_layer(
                        current_layer,
                        current_state,
                        layer_index,
                        dilation,
                        self.residual_channels,
                        self.dilation_channels)
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = tf.Variable(tf.truncated_normal(
                [1, self.residual_channels,
                 int(self.quantization_channels / 2)],
                stddev=0.3,
                name="postprocess1"))
            w2 = tf.Variable(tf.truncated_normal(
                [1, int(self.quantization_channels / 2),
                 self.quantization_channels],
                stddev=0.3,
                name="postprocess2"))

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.matmul(transformed1, w1[0, :, :])
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.matmul(transformed2, w2[0, :, :])

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(input_batch, depth=self.quantization_channels,
                                 dtype=tf.float32)
            if self.fast_generation:
                shape = [self.batch_size, self.quantization_channels]
            else:
                shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(
                encoded, shape)
        return encoded

    def predict_proba(self, waveform, name='wavenet'):
        '''Computes the probability distribution of the next sample.'''
        with tf.variable_scope(name):
            encoded = self._one_hot(waveform)
            if self.fast_generation:
                raw_output = self._create_generator(encoded)
            else:
                raw_output = self._create_network(encoded)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            proba = tf.nn.softmax(tf.cast(out, tf.float64))
            last = tf.slice(proba,
                            [tf.shape(proba)[0] - 1, 0],
                            [1, self.quantization_channels])
            return tf.reshape(last, [-1])
        
    def loss(self, input_batch, name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.variable_scope(name):
            input_batch = mu_law_encode(input_batch,
                                        self.quantization_channels)
            encoded = self._one_hot(input_batch)
            raw_output = self._create_network(encoded)

            with tf.name_scope('loss'):
                # Shift original input left by one sample, which means that
                # each output sample has to predict the next input sample.
                shifted = tf.slice(encoded, [0, 1, 0],
                                   [-1, tf.shape(encoded)[1] - 1, -1])
                shifted = tf.pad(shifted, [[0, 0], [0, 1], [0, 0]])

                prediction = tf.reshape(raw_output,
                                        [-1, self.quantization_channels])
                loss = tf.nn.softmax_cross_entropy_with_logits(
                    prediction,
                    tf.reshape(shifted, [-1, self.quantization_channels]))
                reduced_loss = tf.reduce_mean(loss)

                tf.scalar_summary('loss', reduced_loss)

        return reduced_loss
