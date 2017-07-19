import os
import re
import midi
import random
import librosa
import fnmatch
import threading
import numpy as np
import tensorflow as tf

from newAudioReader import *

DATA_DIR = "/projectnb/textconv/WaveNet/Datasets/unit_test"
LC_FILEFORMAT = "*.mid"

if __name__ == __main__:

	load_files(DATA_DIR, 16000, False, True, LC_FILEFORMAT)