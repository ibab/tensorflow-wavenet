import os
import re
import midi
import random
import librosa
import fnmatch

from newAudioReader import find_files, load_files, clean_midi_files, trim_silence, AudioReader, MidiMapper

DATA_DIR = "/projectnb/textconv/WaveNet/Datasets/unit_test"
LC_FILEFORMAT = "*.mid"

if __name__ == '__main__':
	print("hi :(")

	load_files(DATA_DIR, 16000, False, True, LC_FILEFORMAT)
