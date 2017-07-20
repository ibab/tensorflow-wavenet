import os
import re
import midi
import random
import librosa
import fnmatch

from wavenet import find_files, load_files, clean_midi_files, trim_silence, AudioReader, MidiMapper

TEST_DATA_DIR = "/projectnb/textconv/WaveNet/Datasets/unit_test"
LC_FILEFORMAT = "*.mid"


class AudioReaderTest(tf.test.TestCase):

	def load_file_test(self):
		''' expects two midi and two audio files named the same file name and 
			cheks if the load files works as expected '''
		with self.test_session():
			# set groud truths
			expected_audio = []
			expected_filename = ['testfile1', 'testfile2']
			expected_gc_id = [None, None]
			expected_lc_timeseries = []

			# load_files yilds a generator
			iterator = load_files(TEST_DATA_DIR, 16000, False, True, LC_FILEFORMAT)

			# query said generator and check all values
			index_counter = 0
			for audio, filename, gc_id, lc_timeseries in iterator:

				tf.assertAllEqual(len(audio), expected_audio_len[index_counter])
				tf.assertTrue(filename is expected_filename[index_counter])
				tf.assertTrue(gc_id is expected_gc_id[index_counter])
				tf.assertTrue(lc_timeseries is expected_lc_timeseries[index_counter])
				index_counter++


if __name__ == '__main__':
	tf.test.main()
