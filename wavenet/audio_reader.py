import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename


def load_vctk_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        yield audio, speaker_id


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 batch_size=1,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=256):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.silence_threshold = silence_threshold        

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(audio_dir):
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

        self.read()

    def dequeue(self):
        return self.source 
        # output = self.queue.dequeue_many(num_elements)
        # return output

    def read(self):
        data_ = []
        buffer_ = np.array([])        
        
        iterator = load_generic_audio(self.audio_dir, self.sample_rate)
        for audio, filename in iterator:
            if self.silence_threshold is not None:
                # Remove silence
                audio = trim_silence(audio[:, 0], self.silence_threshold)
                audio = audio.reshape(-1)
                if audio.size == 0:
                    print("Warning: {} was ignored as it contains only "
                            "silence. Consider decreasing trim_silence "
                            "threshold, or adjust volume of the audio."
                            .format(filename))

            if self.sample_size:
                # Cut samples into fixed size pieces
                buffer_ = np.append(buffer_, audio)
                while len(buffer_) > self.sample_size:
                    a = np.reshape(buffer_[:self.sample_size], [-1])                    
                    data_.append(tf.convert_to_tensor(a,dtype=np.float32))
                    buffer_ = buffer_[self.sample_size:]
            else:
                data_.append(tf.convert_to_tensor(audio))
        source = tf.train.slice_input_producer(data_)
        source = tf.train.shuffle_batch([source], batch_size=self.batch_size, num_threads=4, capacity=50000, min_after_dequeue=10000)
        self.source = source        


