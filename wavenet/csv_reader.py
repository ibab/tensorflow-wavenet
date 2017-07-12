# CsvReader without gc implementation.
import os
import re
import fnmatch
import threading
import tensorflow as tf
import multiprocessing

FILE_PATTERN = r'([0-9]+).*\.csv'
FIND_FILES_PATTERN = '*.csv'
# naming of variables

def get_gc(file):
    # need to be changed
    id_reg_expression = re.compile(FILE_PATTERN)
    matches = id_reg_expression.findall(file)[0]
    return int(matches)

def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for _ in range(len(files)):
        matches = id_reg_expression.findall(os.path.basename(files[_]))[0]
        id = int(matches)
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id
    return min_id, max_id

class CsvReader(object):
    def __init__(self, files, receptive_field, sample_size, data_dim=77, gc_enabled=None):
        self.data_batch = self.input_batch(files, data_dim, shuffle=False, batch_size=receptive_field+sample_size)


        #TODO: GC
        if gc_enabled:
            # need to find better implementation for this.
            _, self.gc_category_cardinality = get_category_cardinality(files)
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

        #TODO: LC


    def input_batch(self,
                 filenames,
                 data_dim,
                 num_epochs=None,
                 shuffle=True,
                 skip_header_lines=0,
                 batch_size=200):
      """Generates an input function for training or evaluation.
      This uses the input pipeline based approach using file name queue
      to read data so that entire data is not loaded in memory.
      Args:
          filenames: [str] list of CSV files to read data from.
          num_epochs: int how many times through to read the data.
            If None will loop through data indefinitely
          shuffle: bool, whether or not to randomize the order of data.
            Controls randomization of both file order and line order within
            files.
          skip_header_lines: int set to non-zero in order to skip header lines
            in CSV files.
          batch_size: int First dimension size of the Tensors returned by
            input_fn
      Returns:
          A function () -> (features, indices) where features is a dictionary of
            Tensors, and indices is a single Tensor of label indices.
      """
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
