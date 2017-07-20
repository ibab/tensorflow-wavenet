import os
import re
import midi
import random
import librosa
import fnmatch

from lc_audio_reader import find_files, load_files, clean_midi_files, trim_silence, AudioReader, MidiMapper

TEST_DATA_DIR = "/projectnb/textconv/WaveNet/Datasets/unit_test"
LC_FILEFORMAT = "*.mid"


class AudioReaderTest(tf.test.TestCase):

	def load_file_test(self):
		''' expects two midi and two audio files named the same file name and 
			checks if the load files works as expected '''
		with self.test_session():
			# set ground truths
			expected_audio_len = [18, 19]
			expected_filename = ['/projectnb/textconv/WaveNet/Datasets/unit_test/mond_1_format0', '/projectnb/textconv/WaveNet/Datasets/unit_test/ty_juli_format0']
			expected_gc_id = [None, None]
			expected_lc_timeseries = [3347, 2679]

			# load_files yields a generator
			iterator = load_files(TEST_DATA_DIR, 16000, False, True, LC_FILEFORMAT)

			# query generator and check all values
			index_counter = 0
			for audio, filename, gc_id, lc_timeseries in iterator:

				# Note: files are randomized, so we cannot check sequentially if they match or not
				# Luckily, python is basically English

				# check audio length
				audio_len = len(audio)
				if audio_len in expected_audio_len:
					ind = expected_audio_len.index(audio_len)
					tf.assertTrue(audio_len is expected_audio_len[ind])
				else:
					print("Length of audio file {} not expected.".format(index_counter))

				# check file name
				if filename in expected_filename:
					ind = expected_filename.index(filename)
					tf.assertTrue(filename is expected_filename[ind])
				else:
					print("Filename of audio file {} not expected.".format(index_counter))

				# gc_id should be none, randomized or not
				tf.assertTrue(gc_id is expected_gc_id[index_counter])

				# check midi output length
				lc_length = len(lc_timeseries[0])
				if lc_length in expected_lc_timeseries:
					ind = expected_lc_timeseries.index(lc_length)
					tf.assertTrue(lc_timeseries is expected_lc_timeseries[ind])
				else:
					print("Length of MIDI timeseries {} not expected.".format(index_counter))

				index_counter += 1


if __name__ == '__main__':
	tf.test.main()
