import os
import re
import midi
import random
import librosa
import fnmatch
import threading
import numpy as np
import tensorflow as tf

# from newAudioReader import *

DATA_DIR = "/projectnb/textconv/WaveNet/Datasets/unit_test"
LC_FILEFORMAT = "*.mid"

def find_files(dir, format):
	'''Recursively finds all files matching the pattern.'''
	print("Is anything happening???")
	files = []
	for root, dirnames, filenames in os.walk(directory):
		for filename in fnmatch.filter(filenames, pattern):
			files.append(os.path.join(root, filename))
			print("Files found.")
	return files


def load_files(data_dir, sample_rate, gc_enabled, lc_enabled, lc_fileformat):
	print("is anything working")
	# get all audio files and print their number
	audio_files = find_files(data_dir, '*.wav')
	print("Number of audio files is {}".format(len(audio_files)))

	if lc_enabled:
		lc_files = find_files(data_dir, lc_fileformat)
		print("Number of midi files is {}".format(len(lc_files)))

		# Now make sure the files correspond and are in the same order
		audio_files, lc_files = clean_midi_files(audio_files, lc_files)
		print("File clean up done. Final file count is {}".format(len(audio_files)))

	randomized_files = randomize_files(audio_files)
	for filename in randomized_files:
		# get GC embedding here if using it

		# now load audio file using librosa, audio is now a horizontal array of float32s
		# throwaway _ is the sample rate returned back
		audio, _ = librosa.load(filename, sr = sample_rate, mono = True)
		
		# this reshape makes it a vertical array
		audio = audio.reshape(-1, 1)

		# TODO: This is where we get the GC ID mapping from audio
		# later, we can add support for conditioning on genre title, etc.
		# gc_id = get_gc_id(filename)

		# now we get the LC timeseries file here
		# load in the midi or any other local conditioning file
		if lc_enabled:
			midi_name = os.path.splitext(filename)[0] + ".mid"
			# returns list of events with ticks in relative time
			lc_timeseries = midi.read_midifile(midi_name)

		yield audio, filename, gc_id, lc_timeseries


def clean_midi_files(audio_files, lc_files):
	# mapping both lists of files to lists of strings to compare them
	# note: in Python 3 map() returns a map object, which can still be iterated through (list() not needed)
	str_audio = map(str, audio_files)
	str_midi = map(str, lc_files)

	# remove extensions
	for wav in enumerate(str_audio):
		str_audio[wav] = os.path.splitext(str_audio(wav))[0]

	for midi in enumerate(str_midi):
		str_midi[midi] = os.path.splitext(str_midi(midi))[0]

	# create two lists of the midi and wav mismatches
	str_midi = [midi for midi in str_midi if midi not in str_audio]
	str_audio = [wav for wav in str_audio if wav not in str_midi]

	for wav in str_audio:
		fname = wav + ".wav"
		audio_files.remove(fname)
		print("No MIDI match found for .wav file {}. Raw audio file removed.".format(fname))

	for midi in str_midi:
		fname = midi + ".mid"
		lc_files.remove(fname)
		print("No raw audio match found for .mid file {}. MIDI file removed.".format(fname))
		
	return audio_files, lc_files


if __name__ == '__main__':
	print("hi")

load_files(DATA_DIR, 16000, False, True, LC_FILEFORMAT)
