"""Unit tests for fp16 operation"""

from __future__ import print_function
import json

import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, time_to_batch, batch_to_time, causal_conv


class TestFp16(tf.test.TestCase):
    def testMatMul(self):
        mat1 = np.identity(5)
        mat2 = np.identity(5)
        mat2[1, 1] = 0.23456789123456
        mat2[2, 2] = 0.33333333333333
        mat2[3, 3] = 0.66666666666666
        mat1_fp16 = tf.convert_to_tensor(mat1, dtype=tf.float16)
        mat2_fp16 = tf.convert_to_tensor(mat2, dtype=tf.float16)
        mat3_fp16 = mat1_fp16*mat2_fp16

        with self.test_session() as sess:
            result = sess.run(mat3_fp16)
        print("result: ", result)

if __name__ == '__main__':
    tf.test.main()
