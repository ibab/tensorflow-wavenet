# CsvReader without gc implementation.
import os
import re
import fnmatch
import threading
import tensorflow as tf
import multiprocessing

class CsvReader(object):
    def __init__(self, files, receptive_field, sample_size, config):

        # We use batch size in tf.train.batch to indicate one chunk of data.
        # TODO: Implement a second queue around these blocks if needed.
        batch_size = receptive_field+sample_size

        # Initialize the main data batch. This uses raw values, no lookup table.
        self.data_batch = self.input_batch(files, config["data_dim"], batch_size=batch_size)


        if emotion_suffix:
            emotion_dim = config["emotion_dim"]
            emotion_categories = config["emotion_categories"]

            self.emotion_cardinality = len(emotion_categories)
            self.gc_batch = self.input_batch([i+config["emotion_suffix"] for i in files],
                                                  emotion_dim,
                                                  batch_size=batch_size,
                                                  mapping_strings=emotion_categories)

        if phoneme_suffix:
            phoneme_dim = config["phoneme_dim"]
            phoneme_categories = config["phoneme_categories"]

            self.phoneme_cardinality = len(phoneme_categories)
            self.lc_batch = self.input_batch([i+config["phoneme_suffix"] for i in files],
                                                  phoneme_dim,
                                                  batch_size=batch_size,
                                                  mapping_strings=emotion_categories)



    def input_batch(self,
                 filenames,
                 data_dim,
                 num_epochs=None,
                 skip_header_lines=0,
                 batch_size=200,
                 mapping_strings=None):

      filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
      reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

      _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

      # Parse the CSV File
      record_defaults = [[1.0] for _ in range(data_dim)]
      features = tf.decode_csv(rows, record_defaults=record_defaults)

      # Mapping for Conditioning Files, replace String by lookup table.
      if mapping_strings:
          table = tf.contrib.lookup.index_table_from_tensor(tf.constant(mapping_strings))
          features = table.lookup(features)

      # This operation builds up a buffer of parsed tensors, so that parsing
      # input data doesn't block training
      # If requested it will also shuffle
      features = tf.train.batch(
          features,
          batch_size,
          capacity=batch_size * 10,
          num_threads=multiprocessing.cpu_count(),
          enqueue_many=True,
          allow_smaller_final_batch=False
      )

      return features
