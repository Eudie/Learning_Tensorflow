# Author : Eudie
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import time

import numpy as np
import tensorflow as tf

import reader

# Defining flags
flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

# Reading raw data
raw_data = reader.ptb_raw_data(FLAGS.data_path)
train_path = os.path.join(FLAGS.data_path, "ptb.train.txt")
valid_path = os.path.join(FLAGS.data_path, "ptb.valid.txt")
test_path = os.path.join(FLAGS.data_path, "ptb.test.txt")

with tf.gfile.GFile(train_path, "r") as f:
    data = f.read().decode("utf-8").replace("\n", "<eos>").split()

counter = collections.Counter(data)
count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

words, _ = list(zip(*count_pairs))
word_to_id = dict(zip(words, range(len(words))))

train_data = _file_to_word_ids(train_path, word_to_id)
valid_data = _file_to_word_ids(valid_path, word_to_id)
test_data = _file_to_word_ids(test_path, word_to_id)
vocabulary = len(word_to_id)
