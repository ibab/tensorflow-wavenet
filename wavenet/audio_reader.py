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
        yield audio, None, None


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
        yield audio, speaker_id, None


def trim_silence(audio, threshold, local_features=None):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio_piece = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
    local_features_piece = local_features[indices[0]:
                                          indices[-1] if indices.size
                                          else local_features[0:0]]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio_piece, local_features_piece


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=256):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.threads = []
        self.sample_placeholder = \
            tf.placeholder(dtype=tf.float32, shape=None, name='audio')
        self.global_feature_placeholder = \
            tf.placeholder(dtype=tf.float64, shape=None, name='global_feature')
        self.local_features_placeholder = \
            tf.placeholder(dtype=tf.float64, shape=None, name='local_features')
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'float64', 'float64'],
                                         shapes=[(None, 1), (), (None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder,
                                           self.global_feature_placeholder,
                                           self.local_features_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(audio_dir):
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            # iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            iterator = load_vctk_audio(self.audio_dir, self.sample_rate)
            for audio, global_feature, local_features in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if not global_feature:
                    global_feature = 0.0
                if not local_features:
                    local_features = np.zeros_like(audio)
                if self.silence_threshold is not None:
                    # Remove silence
                    audio, local_features = \
                        trim_silence(audio[:, 0],
                                     self.silence_threshold,
                                     local_features)
                    audio = audio.reshape(-1, 1)
                    local_features = local_features.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    for start_index in range(0, len(audio), self.sample_size):
                        audio_piece = audio[start_index: start_index +
                                            self.sample_size]
                        lf_piece = local_features[start_index: start_index +
                                                  self.sample_size]
                        sess.run(
                            self.enqueue,
                            feed_dict={
                               self.sample_placeholder: audio_piece,
                               self.global_feature_placeholder: global_feature,
                               self.local_features_placeholder: lf_piece,
                            }
                            )
                else:
                    sess.run(
                            self.enqueue,
                            feed_dict={
                               self.sample_placeholder: audio,
                               self.global_feature_placeholder: global_feature,
                               self.local_features_placeholder: local_features,
                               }
                            )

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
