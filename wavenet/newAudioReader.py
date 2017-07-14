import os
import re
import threading
import tensorflow as tf
import numpy as np
import librosa
import random
import fnmatch

def find_files(dir, format):
	'''Recursively finds all files matching the pattern.'''
	files = []
	for root, dirnames, filenames in os.walk(directory):
		for filename in fnmatch.filter(filenames, pattern):
			files.append(os.path.join(root, filename))
	return files

def load_files(data_dir, sample_rate, gc_enabled, lc_enabled):
	# get all audio files and print their number
	audio_files = find_files(data_dir, '*.wav')
	print("Number of audio files is {}".format(len(audio_files)))

	if lc_enabled:
		midi_files = find_files(data_dir, '*.mid')
		print("Number of midi files is {}".format(len(midi_files)))

		# Now make sure the files correspond and are in the same order
		audio_files, midi_files = clean_files(audio_files, midi_files)
		print("File clean up done. Final file count is {}".format(len(audio_files)))

	randomized_files = randomize_files(audio_files)
	for filename in randomized_files:
		# get GC embedding here if using it

		# now load audio file using librosa, audio is now a horizontal array of float32s
		# Throwawav _ is the sample rate returned back
		audio, _ = librosa.load(filename, sr = sample_rate, mono = True)
		
		# this reshape makes it a vertical array
		audio = audio.reshape(-1, 1)

		# TODO: This is where we get the GC ID mapping from audio 
		# gc_id = get_gc_id(audio)

		# now we get the LC timeseries file here
		if lc_enabled:
			# TODO: This is where we load in the midi or any other local conditioning file
			# lc_timeseries = get_lc_file(filename)

		yield audio, filename, gc_id, lc_timeseries

def clean_files(audio_files, midi_files):
	# mapping both lists of files to lists of strings to compare them
	# note: in Python 3 map() returns a map object, which can still be iterated through (list() not needed)
	str_audio = map(str, audio_files)
	str_midi = map(str, midi_files)

	# remove extensions
	for wav in enumerate(str_audio):
		str_audio(wav) = os.path.splitext(str_audio(wav))[0]

	for midi in enumerate(str_midi):
		str_midi(midi) = os.path.splitext(str_midi(midi))[0]

	# create two lists of the midi and wav mismatches
	str_midi = [midi for midi in str_midi if midi not in str_audio]
	str_audio = [wav for wav in str_audio if wav not in str_midi]

	for midi in str_midi:
		rem = audio_files.pop(eval(midi + ".wav"))
		print("No MIDI match found for .wav file {}. File removed.".format(rem))

	for wav in str_audio:
		rem = midi_files.pop(eval(wav + ".mid"))
		print("No raw audio match found for .mid file {}. File removed.".format(rem))

def trim_silence(audio, threshold, frame_length = 2048):
	'''Removes silence at the beginning and end of a sample.'''
	if audio.size < frame_length:
		frame_length = audio.size
	energy = librosa.feature.rmse(audio, frame_length=frame_length)
	frames = np.nonzero(energy > threshold)
	indices = librosa.core.frames_to_samples(frames)[1]

	# Note: indices can be an empty array, if the whole audio was silence.
	return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

class AudioReader():
	def __init__(self,
				 data_dir,
				 coord,
				 sample_rate = 16000,
				 gc_enabled = False,
				 lc_enabled = False,
				 receptive_field,
				 sample_size = None,
				 silence_threshold = None,
				 q_size = 32):
		# Input member vars initialiations
		self.data_dir = data_dir
		self.coord = coord
		self.sample_rate = sample_rate
		self.gc_enabled = gc_enabled
		self.lc_enabled = lc_enabled
		self.receptive_field = receptive_field
		self.sample_size = sample_size
		self.silence_threshold = silence_threshold
		self.q_size = q_size

		# Non-input member vars initialization
		self.threads = []
		
		# DATA QUEUES

		# Audio sampels are float32s with encoded as a one hot, so shape is 1 X quantization_channels
		self.audio_placeholder = tf.placeholder(dtype = tf.float32, shape = None)
		self.q_audio = tf.PaddingFIFOQueue(capacity = q_size, dtypes = [tf.float32], shapes = [(None, 1)])
		self.enq_audio = self.q_audio.enqueue([self.audio_placeholder])


		if self.gc_enabled:
			# GC samples are embedding vectors with the shape of 1 X GC_channels
			self.gc_placeholder = tf.placeholder(dtype = tf.int32, shape = ())
			self.q_gc = tf.PaddingFIFOQueue(capacity = q_size, dtypes = [tf.int32], shapes = [(None, 1)])
			self.enq_gc = self.q_gc.enqueue([self.gc_placeholder])

		if self.lc_enabled:	
			# LC samples are embedding vectors with the shape of 1 X LC_channels
			self.lc_placeholder = tf.placeholder(dtype = tf.int32, shape = None)
			self.q_lc = tf.PaddingFIFOQueue(capacity = q_size, dtypes = [tf.int32], shapes = [(None, 1)])
			self.enq_lc = self.q_lc.enqueue([self.lc_placeholder])

		# now load in the files and see if they exist
		audio_files = find_files(data_dir, '*.wav')
		if not audio_files:
			raise ValueError("No WAV files found in '{}'.".format(data_dir))
		
		# if LC is enabled, check if local conditioning files exist
		if lc_enabled:
			midi_files = find_files(data_dir, '*.mid')
			if not midi_files:
				raise ValueError("No MIDI files found in '{}'".format(data_dir))

		def dq_audio(self, num_elements):
			return self.q_audio.dequeue_many(num_elements)

		def dq_gc(self, num_elements):
			return self.q_gc.dequeue_many(num_elements)

		def dq_lc(self, num_elements):
			return self.q_lc.dequeue_many(num_elements)

		def input_stream(self, sess):
			stop = False

			# keep looping until traning is done
			while not stop:
				iterator = load_files(self.data_dir, self.sample_rate, self.gc_enabled, self.lc_enabled)

			for audio, filename, gc_id, lc_timeseries in iterator:
				if self.coord.should_stop():
					stop = True
					break

				# TODO: If we remove this silence trimming we can use the randomised queue
				# instead of the padding queue so that we dont have to take care of midi with silence
				if self.silence_threshold is not None:
					audio = trim_silence(audio[:, 0], self.silence_threshold)
					audio = audio.reshape(-1, 1)

					# now check if the whole audio was trimmed away
					if audio.size = 0:
						print("Warning: {} was ignored as it contains only "
							  "silence. Consider decreasing trim_silence "
							  "threshold, or adjust volume of the audio."
							  .format(filename))

					# now pad beginning of samples with n = receptive_field number of 0s 
					audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]], 'constant')

					# now 
					if self.sample_size:

					else:
						# otherwise feed the whole audio sample in its entireity
						sess.run(self.enq_audio, feed_dict = {self.audio_placeholder : audio})

						# add GC mapping to q if enabled
						if gc_enabled:
							sess.run(self.enq_gc, feed_dict = {self.gc_placeholder : gc_id})
						
						# add LC mapping to queue if enabled
						if lc_enabled:
							# TODO: this is where the midi gets upsampled and mapped to the wav samples
							# lc = map_midi(audio, lc_timeseries)
							sess.run(self.enq_lc, feed_dict = {self.lc_placeholder : lc})



		def start_threads(self, sess, n_threads=1):
			for _ in range(n_threads):
				thread = threading.Thread(target = self.input_stream, args=(sess,))
				thread.daemon = True  # Thread will close when parent quits.
				thread.start()
				self.threads.append(thread)
			return self.threads
