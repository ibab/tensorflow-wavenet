# CsvReader without gc implementation.
import os
import re
import fnmatch
import threading
import tensorflow as tf

FILE_PATTERN = r'([0-9]+).*\.csv'
FIND_FILES_PATTERN = '*.csv'
# naming of variables

def find_files(file_dir, pattern=FIND_FILES_PATTERN):
    files = []
    for root, dirnames, filenames in os.walk(file_dir):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

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
    def __init__(self, file_dir, data_dim, coord, gc_enabled, receptive_field, sample_size=32, queue_size=250*16):
        self.file_dir = file_dir
        self.coord = coord
        self.gc_enabled = gc_enabled
        self.receptive_field = receptive_field
        self.threads = []
        self.data_dim = data_dim
        self.sample_size = sample_size
        self.queue_size = queue_size

        # Obtain queue of filename
        # tf.train.string_input_producer craetes FIFO queue of tensor object
        self.filename_queue = tf.train.string_input_producer(find_files(self.file_dir))
        self.reader = tf.TextLineReader()

        # Sets up the tf queue to store data
        self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                         ['float32'],
                                         shapes=[(None, self.data_dim)])

        # This is needed for now to make the program go
        if self.gc_enabled:
            self.gc_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
            self.gc_queue = tf.PaddingFIFOQueue(self.queue_size,
                                                ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.gc_placeholder])



            # need to find better implementation for this.
            _, self.gc_category_cardinality = get_category_cardinality(find_files(self.file_dir))
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        return self.queue.dequeue_many(num_elements)

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # while thread is running keep fetching vlaues
        while not stop:
            if self.coord.should_stop():
                stop = True
                break

            # Load dataset
            key, value = self.reader.read_up_to(self.filename_queue, self.receptive_field + self.sample_size)
            record_defaults = [[1.0] for _ in range(self.data_dim)]
            value = tf.decode_csv(value, record_defaults=record_defaults)
            value = tf.transpose(value, [1, 0])

            value = tf.pad(value, [[self.receptive_field, 0], [0, 0]], 'CONSTANT')

            sess.run(self.queue.enqueue(value))

            # Get category_id
            if self.gc_enabled:
                filename = os.path.basename(sess.run(key)[0])
                category_id = filename.split(':')[0]
                category_id = get_gc(category_id)
                sess.run(self.gc_enqueue, feed_dict={self.gc_placeholder: category_id})


    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self.threads
