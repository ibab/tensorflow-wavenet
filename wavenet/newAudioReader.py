import os
import re
import midi
import random
import librosa
import fnmatch
import threading
import numpy as np
import tensorflow as tf
from enum import Enum

def find_files(dir, format):
	'''Recursively finds all files matching the pattern.'''
	files = []
	for root, dirnames, filenames in os.walk(directory):
		for filename in fnmatch.filter(filenames, pattern):
			files.append(os.path.join(root, filename))
	return files

def load_files(data_dir, sample_rate, gc_enabled, lc_enabled, lc_fileformat):
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
		str_audio(wav) = os.path.splitext(str_audio(wav))[0]

	for midi in enumerate(str_midi):
		str_midi(midi) = os.path.splitext(str_midi(midi))[0]

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

	
def map_midi(audio, lc_timeseries)
    '''Upsampling midi and mapping it to the wav samples.'''
    # TODO
    

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
				 lc_fileformat = None,
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
		self.lc_fileformat = lc_fileformat
		self.receptive_field = receptive_field
		self.sample_size = sample_size
		self.silence_threshold = silence_threshold
		self.q_size = q_size

		# Non-input member vars initialization
		self.threads = []
		
		# DATA QUEUES

		# Audio samples are float32s with encoded as a one hot, so shape is 1 X quantization_channels
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
		audio_files = find_files(self.data_dir, '*.wav')
		if not audio_files:
			raise ValueError("No WAV files found in '{}'.".format(data_dir))
		
		# if LC is enabled, check if local conditioning files exist
		if lc_enabled:
			lc_files = find_files(self.data_dir, self.lc_fileformat)
			if not lc_files:
				raise ValueError("No MIDI files found in '{}'".format(self.data_dir))

		def dq_audio(self, num_elements):
			return self.q_audio.dequeue_many(num_elements)

		def dq_gc(self, num_elements):
			return self.q_gc.dequeue_many(num_elements)

		def dq_lc(self, num_elements):
			return self.q_lc.dequeue_many(num_elements)

		def input_stream(self, sess):
			stop = False

			# keep looping until training is done
			while not stop:
				iterator = load_files(self.data_dir, self.sample_rate, self.gc_enabled, self.lc_enabled, self.lc_fileformat)

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
					# TODO: figure out why we are padding this ???
					audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]], 'constant')

					# now 
					if self.sample_size:
						# TODO: understand the reason for this piece voodoo from the original reader
						while len(audio) > self.receptive_field:
							piece = audio[:(self.receptice_field + self.sample_size), :]
							sess.run(self.enq_audio, feed_dict = {self.audio_placeholder : piece})

							# add GC mapping to q if enabled
							if self.gc_enabled:
								sess.run(self.enq_gc, feed_dict = {self.gc_placeholder : gc_id})

							# add LC mapping to queue if enabled
							if self.lc_enabled:
								# TODO:
								# lc = map_midi(piece)
								sess.run(self.enq_lc, feed_dict = {self.lc_placeholder : lc_encode})
					else:
						# otherwise feed the whole audio sample in its entireity
						sess.run(self.enq_audio, feed_dict = {self.audio_placeholder : audio})

						# add GC mapping to q if enabled
						if gc_enabled:
							sess.run(self.enq_gc, feed_dict = {self.gc_placeholder : gc_id})
						
						# add LC mapping to queue if enabled
						if lc_enabled:
							# TODO: this is where the midi gets upsampled and mapped to the wav samples
							# lc = map_midi(audio, start_sample, lc_timeseries)
							sess.run(self.enq_lc, feed_dict = {self.lc_placeholder : lc_encode})



		def start_threads(self, sess, n_threads = 1):
			for _ in range(n_threads):
				thread = threading.Thread(target = self.input_stream, args = (sess,))
				thread.daemon = True  # Thread will close when parent quits.
				thread.start()
				self.threads.append(thread)
			return self.threads



# Template for the midi mapper
class MidiMapper():
    
    def __init__(self,
                start_sample,
                sample_rate,
                end_sample,
                midi):
        
        self.start_sample = start_sample
        self.sample_rate = sample_rate
        self.end_sample = end_sample
        self.midi = midi
        self.tempo, self.ticks_per_beat = get_midi_metadata(self.midi)
        self.events = enum
            
      
    
    def sample_to_millisecond(self, sample_num, sample_rate):
        '''takes in a sample number of the wav and the sample rate and 
            gets the corresponding millisecond of the sample in the song'''
        return 1000 * sample_num / sample_rate
        
        
    def tick_delta_to_milliseconds(self, delta_ticks):
        # converts a range of ticks into a range of milliseconds
        return self.tempo * delta_ticks / self.ticks_per_beat
        
    
    def get_midi_metadata(self):
        # get all the metadata here from the file header
        return tempo, ticks_per_beat
        
    
    def make_midi_mappings(self, midi, sample_rate, tempo, start_sample = 0, end_sample = None):
        
        state = []
        
        # First get the start and end times of the midi section to be extracted and upsampled
        current_time = sample_to_milliseconds(start_sample, sample_rate)
        end_time = sample_to_milliseconds(end_sample, sample_rate)
        
        while current_time is not end_time or midi is not at_end:
            curr_event = get current event # Need to first descend into pattern THEN to track
            
            if   curr_event.name is "Note On" and delta_tick is 0:
                # add note to state list
            elif curr_event.name is "Note On" and delta_ticks is not 0:
                # first add to embeddings and then add note to state list, then update time
            elif curr_event.name is "Note Off" and delta_ticks is 0:
                # take out of state list
            elif curr_event.name is "Note Off" and delta_ticks is not 0:
                # first add to embeddings and then take out of state list, then update time
            elif curr_event.name is "End of Track":
                # pad until end of song or warn if too long, then update time
            else:
                continue

            current_time = current_time + tick_delta_to_milliseconds(delta_ticks)
    
        
