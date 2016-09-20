import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf


def find_files(directory, pattern='*.wav'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    files = find_files(directory)
    for f in files:
        audio, sr = librosa.load(f, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, 0


def load_vctk_audio(directory, sample_rate):
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for j, f in enumerate(files):
        audio, sr = librosa.load(f, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        speaker_id, recording_id = [int(i) for i in speaker_re.findall(f)[0]]
        yield audio, speaker_id


def trim_sample(audio, threshold=0.3):
    '''Removes silence in the beginning and end of a sample'''
    energy = librosa.feature.rmse(audio)
    indices = librosa.core.frames_to_samples(np.nonzero(energy > threshold))[1]
    return audio[indices[0]:indices[-1]]


class AudioReader(object):
    def __init__(self, args, wavenet_params, coord, sample_size=None):
        self.args = args
        self.wavenet_params = wavenet_params
        self.coord = coord
        self.sample_size = sample_size
        self.threads = []
        self.dataX = tf.placeholder(dtype=tf.float32, shape=None)
        self.dataY = tf.placeholder(dtype=tf.int32)
        self.queue = tf.PaddingFIFOQueue(
            256,  # Queue size.
            ['float32', 'int32'],
            shapes=[(None, 1), ()])
        self.enqueue = self.queue.enqueue([self.dataX, self.dataY])

    def get_inputs(self):
        self.audio_batch, _ = self.queue.dequeue_many(self.args.batch_size)
        return self.audio_batch, _

    def thread_main(self, sess):
        buff = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_vctk_audio(
                self.args.data_dir,
                self.wavenet_params["sample_rate"])
            for audio, label in iterator:
                if self.coord.should_stop():
                    self.stop_threads()
                    stop = True
                    break
                # Remove silence
                audio = trim_sample(audio[:,0])
                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buff = np.append(buff, audio)
                    while len(buff) > self.sample_size:
                        piece = np.reshape(buff[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue, feed_dict={self.dataX: piece, self.dataY: label})
                        buff = buff[self.sample_size:]
                else:
                    sess.run(self.enqueue, feed_dict={self.dataX: audio, self.dataY: label})

    def stop_threads():
        for t in self.threads:
            t.stop()

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(t)
        return self.threads
