from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0.1


def get_args():

	def _ensure_positive_float(f):
		"""Ensure argument is a positive float."""
		if float(f) < 0:
			raise argparse.ArgumentTypeError(
					'Argument must be greater than zero')
		return float(f)

	parser = argparse.ArgumentParser(description = 'WaveNet generation script')

	parser.add_argument('--checkpoint',
		type = str,
		help = 'Which model checkpoint to generate from')

	parser.add_argument('--samples',
		type = int,
		default = SAMPLES,
		help = 'How many waveform samples to generate')
	
	parser.add_argument('--temperature',
		type = _ensure_positive_float,
		default = TEMPERATURE,
		help = 'Sampling temperature')
	
	parser.add_argument('--logdir',
		type = str,
		default = LOGDIR,
		help = 'Directory in which to store the logging information for TensorBoard.')
	
	parser.add_argument('--wavenet-params',
		type = str,
		default = WAVENET_PARAMS,
		help = 'JSON file with the network parameters')
	
	parser.add_argument('--wav-out-path',
		type = str,
		default = None,
		help = 'Path to output wav file')
	
	parser.add_argument('--save-every',
		type = int,
		default = SAVE_EVERY,
		help = 'How many samples before saving in-progress wav')
	
	parser.add_argument('--fast-generation',
		type = bool,
		default = True,
		help = 'Use fast generation')
	
	parser.add_argument('--wav-seed',
		type = str,
		default = None,
		help = 'The wav file to start generation from')
	
	parser.add_argument('--gc-channels',
		type = int,
		default = None,
		help = 'Number of global condition channels. Default: None. Expecting: int')

	parser.add_argument('--gc-cardinality',
		type = int,
		default = None,
		help = 'Number of categories upon which we globally condition.')
	
	parser.add_argument('--gc-id',
		type = int,
		default = None,
		help = 'ID of category to generate, if globally conditioned.')

	parser.add_argument('--lc-channels',
		type = int,
		default = None,
		help = "Number of local conditioning channels. Default: None. Expecting: int")

	parser.add_argument('--lc-fileformat',
		type = str,
		default = None,
		help = "Extension of files being used for local conditioning. Default: None. Expecting: string")

	parser.add_argument('--lc-filepath',
	    type = str,
	    default = None,
	    help = "Path to the file to be used for local condition based generation. Default: None. Expecting: string.")

	args = parser.parse_args()

	if args.gc_channels is not None:
		if args.gc_cardinality is None:
			raise ValueError("Globally conditioning but gc-cardinality not specified.")

		if args.gc_id is None:
			raise ValueError("Globally conditioning enalbed but not GC ID specified.")

	if args.lc_channels is not None:
		if args.lc_fileformat is None:
			raise ValueError("Local conditioning enabled with channels but no file format specified.")

		if args.lc_filepath is None:
			raise ValueError("No local conditioning file provided in the LC filepath")

	return args


def write_wav(waveform, sample_rate, filename):
	y = np.array(waveform)
	librosa.output.write_wav(filename, y, sample_rate)
	print('Updated wav file at {}'.format(filename))


def create_seed(filename,
				sample_rate,
				quantization_channels,
				window_size,
				silence_threshold = SILENCE_THRESHOLD):
	audio, _ = librosa.load(filename, sr = sample_rate, mono = True)
	audio = audio_reader.trim_silence(audio, silence_threshold)

	quantized = mu_law_encode(audio, quantization_channels)
	cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
						lambda: tf.size(quantized),
						lambda: tf.constant(window_size))

	return quantized[:cut_index]


def main():
	args = get_args()
	started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
	logdir = os.path.join(args.logdir, 'generate', started_datestring)
	
	with open(args.wavenet_params, 'r') as config_file:
		wavenet_params = json.load(config_file)

	sess = tf.Session()

	net = WaveNetModel(
		batch_size = 1,
		dilations = wavenet_params['dilations'],
		filter_width = wavenet_params['filter_width'],
		residual_channels = wavenet_params['residual_channels'],
		dilation_channels = wavenet_params['dilation_channels'],
		quantization_channels = wavenet_params['quantization_channels'],
		skip_channels = wavenet_params['skip_channels'],
		use_biases = wavenet_params['use_biases'],
		scalar_input = wavenet_params['scalar_input'],
		initial_filter_width = wavenet_params['initial_filter_width'],
		gc_channels = args.gc_channels,
		gc_cardinality = args.gc_cardinality,
		lc_channels = args.lc_channels)

	# TODO: decide where to load the midi file and upsample it using the MIDI mapper
	samples = tf.placeholder(tf.int32)

	# TODO: if we instantiate the mapper here, we would feed the LC embedding here to the prediction function
	if args.fast_generation:
		next_sample = net.predict_proba_incremental(samples, args.gc_id)
	else:
		next_sample = net.predict_proba(samples, args.gc_id)

	if args.fast_generation:
		sess.run(tf.global_variables_initializer())

		# init_ops is inside model.create_generator
		# init_ops = init = q.enqueue_many(tf.zeros((1, self.batch_size, self.quantization_channels)))
		sess.run(net.init_ops)

	# gather all vars to restore
	variables_to_restore = {
		var.name[:-2]: var for var in tf.global_variables()
		if not ('state_buffer' in var.name or 'pointer' in var.name)}
	saver = tf.train.Saver(variables_to_restore)
	
	# restore all vars
	print('Restoring model from {}'.format(args.checkpoint))
	saver.restore(sess, args.checkpoint)

	decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

	quantization_channels = wavenet_params['quantization_channels']

	# if we are local conditioning then we should not need a seed at the beginning
	if args.wav_seed:
		# should not need this for LC
		seed = create_seed(args.wav_seed,
						   wavenet_params['sample_rate'],
						   quantization_channels,
						   net.receptive_field)
		waveform = sess.run(seed).tolist()
	else:
		# Silence with a single random sample at the end.
		waveform = [quantization_channels / 2] * (net.receptive_field - 1)
		waveform.append(np.random.randint(quantization_channels))

	if args.fast_generation and args.wav_seed:
		# When using the incremental generation, we need to
		# feed in all priming samples one by one before starting the
		# actual generation.
		# This could be done much more efficiently by passing the waveform
		# to the incremental generator as an optional argument, which would be
		# used to fill the queues initially.
		outputs = [next_sample]
		outputs.extend(net.push_ops)

		print('Priming generation...')
		for i, x in enumerate(waveform[-net.receptive_field: -1]):
			if i % 100 == 0:
				print('Priming sample {}'.format(i))
			sess.run(outputs, feed_dict={samples: x})
		print('Done.')

	last_sample_timestamp = datetime.now()

	# for each sample to be generated do the ops in the loop
	for step in range(args.samples):
		# this is where it should be changed to account for LC?
		if args.fast_generation:
			outputs = [next_sample]
			# push_ops is inside model's create_generator
			# where push_ops.append(push)
			# where push = q.enqueue([current_layer])
			# where current_layer = input_batch of the input to the create_generator function
			outputs.extend(net.push_ops)
			window = waveform[-1]
		else:
			if len(waveform) > net.receptive_field:
				window = waveform[-net.receptive_field:]
			else:
				window = waveform
			outputs = [next_sample]

		# Run the WaveNet to predict the next sample.
		prediction = sess.run(outputs, feed_dict = {samples : window})[0]

		# this should not need to be changed for LC
		# Scale prediction distribution using temperature.
		np.seterr(divide = 'ignore')
		scaled_prediction = np.log(prediction) / args.temperature
		scaled_prediction = (scaled_prediction -
							 np.logaddexp.reduce(scaled_prediction))
		scaled_prediction = np.exp(scaled_prediction)
		np.seterr(divide = 'warn')

		# Prediction distribution at temperature=1.0 should be unchanged after
		# scaling.
		if args.temperature == 1.0:
			np.testing.assert_allclose(
					prediction, scaled_prediction, atol=1e-5,
					err_msg = 'Prediction scaling at temperature=1.0 '
							'is not working as intended.')

		sample = np.random.choice(
			np.arange(quantization_channels), p = scaled_prediction)
		waveform.append(sample)

		# Show progress only once per second.
		current_sample_timestamp = datetime.now()
		time_since_print = current_sample_timestamp - last_sample_timestamp
		if time_since_print.total_seconds() > 1.:
			print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
				  end = '\r')
			last_sample_timestamp = current_sample_timestamp

		# If we have partial writing, save the result so far.
		if (args.wav_out_path and args.save_every and
				(step + 1) % args.save_every == 0):
			out = sess.run(decode, feed_dict={samples: waveform})
			write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

	# Introduce a newline to clear the carriage return from the progress.
	print()

	# Save the result as an audio summary.
	datestring = str(datetime.now()).replace(' ', 'T')
	writer = tf.summary.FileWriter(logdir)
	tf.summary.audio('generated', decode, wavenet_params['sample_rate'])
	summaries = tf.summary.merge_all()
	summary_out = sess.run(summaries,
						   feed_dict={samples: np.reshape(waveform, [-1, 1])})
	writer.add_summary(summary_out)

	# Save the result as a wav file.
	if args.wav_out_path:
		out = sess.run(decode, feed_dict={samples: waveform})
		write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

	print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
	main()
