import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf


def scan_directory(directory):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.wav'):
            files.append(os.path.join(root, filename))
    return files


def iterate_through_directory(directory, sample_rate):
    files = scan_directory(directory)
    for f in files:
        audio, sr = librosa.load(f, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, 0


def iterate_through_vctk(directory, sample_rate):
    files = scan_directory(directory)
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


class CustomRunner(object):
    def __init__(self, args, wavenet_params, coord):
        self.args = args
        self.wavenet_params = wavenet_params
        self.coord = coord
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
            for audio, label in iterate_through_vctk(self.args.data_dir, self.wavenet_params["sample_rate"]):
                if self.coord.should_stop():
                    self.stop_threads()
                    stop = True
                    break
                # Remove silence
                audio = trim_sample(audio[:,0])
                # Always cut samples into fixed size pieces
                buff = np.append(buff, audio)
                size = 100000
                while len(buff) > size:
                    piece = np.reshape(buff[:size], [-1, 1])
                    sess.run(self.enqueue, feed_dict={self.dataX: piece, self.dataY: label})
                    buff = buff[size:]

    def stop_threads():
        for t in self.threads:
            t.stop()

    def start_threads(self, sess, n_threads=1):
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            self.threads.append(t)
        return self.threads
