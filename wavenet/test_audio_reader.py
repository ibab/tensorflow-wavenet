import os
import re
import midi
import random
import librosa
import fnmatch
import threading
import numpy as np
import tensorflow as tf
import time
from lc_audio_reader import find_files, load_files, randomize_files, clean_midi_files, trim_silence, AudioReader, MidiMapper

TEST_DATA_DIR = "/home/vijay/Desktop/Summer2k17/Research/LCWavenet/"
LC_FILEFORMAT = "*.mid"

def load_file_test():
		'''Expects two midi and two audio files named the same file name and checks if the load files works as expected.'''

		# Set ground truths
		expected_audio_len = [1812828, 5879478, 64007]
		expected_filename = ['/home/vijay/Desktop/Summer2k17/Research/LCWavenet/test1_format0.wav',
							 '/home/vijay/Desktop/Summer2k17/Research/LCWavenet/ty_juli_format0.wav',
							 '/home/vijay/Desktop/Summer2k17/Research/LCWavenet/mond_1_format0.wav']
		expected_gc_id = [None, None, None]
		expected_lc_timeseries = [3347, 2679, 5]

		# load_files yields a generator
		iterator = load_files(TEST_DATA_DIR, 16000, False, True, LC_FILEFORMAT)

		# query generator and check all values
		index_counter = 0
		for audio, filename, gc_id, lc_timeseries in iterator:

			# Note: files are randomized, so we cannot check sequentially if they match or not
			# check audio length
			audio_len = len(audio)
			assert (audio_len in expected_audio_len), "Length of audio file {} not expected.".format(index_counter)

			# check file name
			# assert (filename in expected_filename), "Filename {} of audio file {} not expected.".format(filename, index_counter)

			# gc_id should be none, randomized or not
			assert (gc_id in expected_gc_id), "Unexpected GC ID."

			# check midi output length
			lc_length = len(lc_timeseries[0])
			assert (lc_length in expected_lc_timeseries), "Length of MIDI timeseries {} not expected.".format(index_counter)

			index_counter += 1

		print("Load file test passed.")


class AudioReaderTest(tf.test.TestCase):

	def setUp(self):
		self.coord = tf.train.Coordinator()
		self.threads = None

	def testReader(self):

		with self.test_session() as sess:
			self.reader = AudioReader(data_dir = TEST_DATA_DIR,
							coord = self.coord,
							receptive_field = 5117, #as opposed to 5120
							lc_enabled = True,
							lc_channels = 128,
							lc_fileformat = LC_FILEFORMAT,
							sess = sess)

			dqd_audio = self.reader.dq_audio(100)
			print("Here 1")
			dqd_upsampled_midi = self.reader.dq_lc(100)
			print(dqd_audio)
			print(dqd_upsampled_midi)
			print("Here 2")

			self.threads = tf.train.start_queue_runners(sess = sess, coord = self.coord)
			print("Here 3")
			sess.run(tf.global_variables_initializer())
			print("Here 4")
			self.reader.start_threads()
			print("Here 5")
			time.sleep(10)

			print("TIME'S UP")

			self.coord.request_stop()
			self.coord.join(self.threads)


if __name__ == '__main__':
	load_file_test()
	tf.test.main()
