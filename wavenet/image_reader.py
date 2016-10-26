import fnmatch
import os
import threading

import numpy as np
import tensorflow as tf
from PIL import Image


def find_files(directory, pattern='*.jpg'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def _read_image(filename):
    return Image.open(filename).convert('L')


def load_generic_image(directory, pattern):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory, pattern)
    for filename in files:
        pic = _read_image(filename)
        pic = pic.resize((64, 64), Image.ANTIALIAS)
        img = np.array(pic)
        img = np.array(img, dtype='float32')
        img = img.reshape(-1, 1)
        yield img, filename


class ImageReader(object):
    '''Generic background text reader that preprocesses image files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 image_dir,
                 coord,
                 sample_size=None,
                 queue_size=256,
                 pattern='*.jpg'):
        self.image_dir = image_dir
        self.pattern = pattern
        self.coord = coord
        self.sample_size = sample_size
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        encode_output = tf.cast(output, tf.int32)
        return encode_output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_image(self.image_dir, self.pattern)
            for image, filename in iterator:
                if self.coord.should_stop():
                    self.stop_threads()
                    stop = True
                    break
                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, image)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: image})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
