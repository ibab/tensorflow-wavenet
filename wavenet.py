import tensorflow as tf
from wavenet_ops import causal_conv

class WaveNet(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        quantization_steps = 2**8  # Quantize to 256 possible amplitude values.
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        net = WaveNet(batch_size, channels, dilations, filter_width,
                      residual_channels, dilation_channel)
        loss = net.loss(input_batch)
    '''
    def __init__(self,
                 batch_size,
                 quantization_steps,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            quantization_steps: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
        '''
        self.batch_size = batch_size
        self.quantization_steps = quantization_steps
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels


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
                               dilation_channels):
        '''Creates a single causal dilated convolution layer.'''
        weights_filter = tf.Variable(tf.truncated_normal(
            [self.filter_width, in_channels, dilation_channels],
            stddev=0.2, name="filter"))
        weights_gate = tf.Variable(tf.truncated_normal(
            [self.filter_width, in_channels, dilation_channels],
            stddev=0.2, name="gate"))

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        weights_dense = tf.Variable(tf.truncated_normal(
            [1, dilation_channels, in_channels], stddev=0.2, name="dense"))
        transformed = tf.nn.conv1d(out, weights_dense, stride=1,
                                   padding="SAME", name="dense")
        layer = 'layer{}'.format(layer_index)
        tf.histogram_summary(layer + '_filter', weights_filter)
        tf.histogram_summary(layer + '_gate', weights_gate)
        tf.histogram_summary(layer + '_dense', weights_dense)

        return transformed, input_batch + transformed


    def encode(self, audio):
        '''Quantizes waveform amplitudes.'''
        with tf.name_scope('preprocessing'):
            mu = self.quantization_steps - 1
            # Perform mu-law companding transformation (ITU-T, 1988).
            magnitude = tf.log(1 + mu * tf.abs(audio)) / tf.log(1. + mu)
            signal = tf.sign(audio) * magnitude
            # Quantize signal to the specified number of levels.
            quantized = tf.cast((signal + 1) / 2 * mu, tf.int32)
        return quantized


    def decode(self, output):
        mu = self.channels - 1
        y = tf.cast(output, tf.float32)
        y = 2 * (output / mu) - 1
        x = tf.sign(y) * (1 / mu) * ((1 + mu)**abs(y) - 1)
        return x


    def _create_network(self, input_batch):
        '''Creates a WaveNet network.'''
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        current_layer = self._create_causal_layer(current_layer,
                                                  self.quantization_steps,
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
                        self.dilation_channels)
                    outputs.append(output)

        # Create the output layer that aggregates the skip connections.
        output_layer = self._create_output_layer(sum(outputs))
        return output_layer


    def _create_output_layer(self, input_tensor):
        '''Processes the input through a convolution stack.'''
        with tf.name_scope('postprocessing'):
            # Perform ReLU -> 1x1 conv -> ReLU -> 1x1 conv to postprocess the
            # output.
            layer_channels = [self.residual_channels,
                              int(self.quantization_steps / 2),
                              self.quantization_steps]
            zipped_channels = zip(layer_channels[:-1], layer_channels[1:])
            next_input = input_tensor
            for layer_index, channels in enumerate(zipped_channels):
                prev_channels, next_channels = channels[:]
                name = "postprocess{}".format(layer_index)
                weights = tf.Variable(tf.truncated_normal(
                    [1, prev_channels, next_channels],
                    stddev=0.3, name=name))
                tf.histogram_summary(name + '_weights', weights)
                relu = tf.nn.relu(next_input)
                next_input = tf.nn.conv1d(relu, weights, stride=1,
                                          padding="SAME")
            return next_input


    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(input_batch, depth=self.quantization_steps,
                                 dtype=tf.float32)
            encoded = tf.reshape(
                encoded, [self.batch_size, -1, self.quantization_steps])
            return encoded


    def predict_proba(self, waveform, name='wavenet'):
        '''Computes the probability distribution of the next sample.'''
        with tf.variable_scope(name):
            encoded = self._one_hot(waveform)
            raw_output = self._create_network(encoded)
            out = tf.reshape(raw_output, [-1, self.quantization_steps])
            proba = tf.nn.softmax(tf.cast(out, tf.float64))
            last = tf.slice(proba,
                            [tf.shape(proba)[0] - 1, 0],
                            [1, self.quantization_steps])
            return tf.reshape(last, [-1])


    def loss(self, input_batch, name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.variable_scope(name):
            mu_law_encoded = self.encode(input_batch)  # Mu-law encoding.
            one_hot = self._one_hot(mu_law_encoded)
            raw_output = self._create_network(one_hot)
            loss = self._create_loss(raw_output, one_hot)
            return loss


    def _create_loss(self, prediction, truth):
        '''Computes the loss given the input and the reconstruction.

        This is an auto-encoding loss computed using the cross-entropy of the
        predicted values.
        '''
        with tf.name_scope('loss'):
            # Shift original input left by one sample, which means that
            # each output sample has to predict the next input sample.
            shifted = tf.slice(truth, [0, 1, 0],
                               [-1, tf.shape(truth)[2] - 1, -1])
            shifted = tf.pad(shifted, [[0, 0], [0, 1], [0, 0]])
            prediction = tf.reshape(prediction, [-1, self.quantization_steps])
            loss = tf.nn.softmax_cross_entropy_with_logits(
                prediction,
                tf.reshape(shifted, [-1, self.quantization_steps]))
            # Compute average loss across all samples and examples.
            reduced_loss = tf.reduce_mean(loss)
            tf.scalar_summary('loss', reduced_loss)
            return reduced_loss
