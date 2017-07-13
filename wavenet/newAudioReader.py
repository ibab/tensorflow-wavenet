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
		audio_files, midi_files = order_midi_files(audio_files, midi_files)

	# now we have all the audio files and the midi files
	# first, check if all the files in the audio have a corresponding midi file
	# then, if they do, order then correctly in one to one order

def order_midi_files(audio_files, midi_files):
	for audio_file in audio_files:
		# get name of audio file
		# check if that file name is in midi file
		# if it is keep it and then put the midi in the order
		# if not remove audio and and then print error message
		# 


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

		def thread(self, sess):
			stop = False

			# keep looping until traning is done
			while not stop:
				iterator = load_files(self.data_dir, self.sample_rate)
