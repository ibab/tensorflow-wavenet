# CsvReader without gc implementation.
import os
import re
import fnmatch
import threading
import tensorflow as tf
import multiprocessing

class CsvReader(object):
    def __init__(self, files, receptive_field, sample_size, data_dim=77, emotion_suffix=None, phoneme_suffix=None):

        batch_size = receptive_field+sample_size

        self.data_batch = self.input_batch(files, data_dim, shuffle=False, batch_size=batch_size)

        if emotion_suffix:
            self.emotion_batch = self.input_batch([i+emotion_suffix for i in files], data_dim, shuffle=False, batch_size=batch_size)

        if phoneme_suffix:
            self.phoneme_batch = self.input_batch([i+phoneme_suffix for i in files], data_dim, shuffle=False, batch_size=batch_size)


    def input_batch(self,
                 filenames,
                 data_dim,
                 num_epochs=None,
                 shuffle=True,
                 skip_header_lines=0,
                 batch_size=200):

      filename_queue = tf.train.string_input_producer(
          filenames, num_epochs=num_epochs, shuffle=shuffle)
      reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

      _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

      # Parse the CSV File
      record_defaults = [[1.0] for _ in range(data_dim)]
      features = tf.decode_csv(rows, record_defaults=record_defaults)

      # This operation builds up a buffer of parsed tensors, so that parsing
      # input data doesn't block training
      # If requested it will also shuffle
      if shuffle:
        features = tf.train.shuffle_batch(
            features,
            batch_size,
            min_after_dequeue=2 * batch_size + 1,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )
      else:
        features = tf.train.batch(
            features,
            batch_size,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=True
        )

      return features
