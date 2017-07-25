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
        batch_size = receptive_field + sample_size
        self.batch_size = batch_size

        self.data_dim = config["data_dim"]

        # Initialize the main data batch. This uses raw values, no lookup table.
        data_files = [files[i] for i in xrange(len(files)) if files[i].endswith(config["data_suffix"])]

        self.data_batch = self.input_batch(data_files, config["data_dim"], batch_size=batch_size)

        if config["emotion_enabled"]:
            emotion_dim = config["emotion_dim"]
            emotion_categories = config["emotion_categories"]
            emotion_files = [files[i] for i in xrange(len(files)) if files[i].endswith(config["emotion_suffix"])]

            self.emotion_cardinality = len(emotion_categories)
            self.gc_batch = self.input_batch(emotion_files,
                                             emotion_dim,
                                             batch_size=batch_size,
                                             mapping_strings=emotion_categories)

        if config["phoneme_enabled"]:
            phoneme_dim = config["phoneme_dim"]
            phoneme_categories = config["phoneme_categories"]
            phoneme_files = [files[i] for i in xrange(len(files)) if files[i].endswith(config["phoneme_suffix"])]

            self.phoneme_cardinality = len(phoneme_categories)
            self.lc_batch = self.input_batch(phoneme_files,
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

        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
        reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

        _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

        # Parse the CSV File
        if mapping_strings:
            default_value = "N/A"
        else:
            default_value = 1.0

        record_defaults = [[default_value] for _ in range(data_dim)]
        features = tf.decode_csv(rows, record_defaults=record_defaults)

        # Mapping for Conditioning Files, replace String by lookup table.
        if mapping_strings:
            table = tf.contrib.lookup.index_table_from_tensor(tf.constant(mapping_strings))
            features = table.lookup(tf.stack(features))
            features = tf.unstack(features)

        # This operation builds up a buffer of parsed tensors, so that parsing
        # input data doesn't block training
        # If requested it will also shuffle
        features = tf.train.batch(
            features,
            batch_size,
            capacity=batch_size * 100,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=True,
            allow_smaller_final_batch=False
        )

        return features
