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
		audio_files, midi_files = order_midi_files(audio_files, midi_files)
		print("File clean up done. Final file count is {}".format(len(audio_files)))

		randomized_files = randomize_files(audio_files)
		for filename in randomized_files:
			# get GC embedding here if using it

			# now load audio file using librosa
			audio, _ = librosa.load(filename, sr = sample_rate, mono = True)
			
			# reshape single channel for broadcast
			audio = audio.reshape(-1, 1)

			# now we get the LC timeseries file here
			lc_file = get_lc_file(filename)

			yield audio, filename, midi_file

	# now we have all the audio files and the midi files
	# first, check if all the files in the audio have a corresponding midi file
	# then, if they do, order then correctly in one to one order

def order_midi_files(audio_files, midi_files):
	midi_ind = []
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
		rem = audio_files.pop(str(midi + ".wav"))
		print("No MIDI match found for .wav file {}. File removed.".format(rem))

	for wav in str_audio:
		rem = midi_files.pop(str(wav + ".mid"))
		print("No raw audio match found for .mid file {}. File removed.".format(rem))

	# now we have two lists of the same files
	# sort list of midi_files according to the audio_files order


def trim_silence(audio, threshold, frame_length=2048):
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
		self.sample_placeholder = tf.placeholder(dtype = tf.float32, shape = None)
		
		# DATA QUEUES

		# Audio sampels are float32s with encoded as a one hot, so shape is 1 X quantization_channels
		self.q_audio = tf.PaddingFIFOQueue(capacity = q_size, dtypes = [tf.float32], shapes = [(None, 1)])

		if self.gc_enabled:
			# GC samples are embedding vectors with the shape of 1 X GC_channels
			self.q_gc = tf.PaddingFIFOQueue(capacity = q_size, dtypes = [tf.int32], shapes = [(None, 1)])

		if self.lc_enabled: 	
			# LC samples are embedding vectors with the shape of 1 X LC_channels
			self.q_lc = tf.PaddingFIFOQueue(capacity = q_size, dtypes = [tf.int32], shapes = [(None, 1)])


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
				iterator = load_files(self.data_dir, self.sample_rate)

			for 

		def start_threads(self, sess, n_threads=1):
			for _ in range(n_threads):
				thread = threading.Thread(target = self.input_stream, args=(sess,))
				thread.daemon = True  # Thread will close when parent quits.
				thread.start()
				self.threads.append(thread)
			return self.threads
