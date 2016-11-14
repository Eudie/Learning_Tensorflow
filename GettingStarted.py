#!/usr/bin/python

import tensorflow as tf
import time


with tf.device('/cpu:0'):
    x1 = tf.Variable(tf.random_normal([11111, 11111]))
    x2 = tf.Variable(tf.random_normal([11111, 11111]))
    result = tf.matmul(x1, x2)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        start_time_1 = time.time()
        output = sess.run([result])
        duration_1 = time.time() - start_time_1
        print "CPU", duration_1

with tf.device('/gpu:0'):
    x3 = tf.Variable(tf.random_normal([11111, 11111]))
    x4 = tf.Variable(tf.random_normal([11111, 11111]))
    result1 = tf.matmul(x3, x4)
    init1 = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init1)
        start_time_2 = time.time()
        output1 = sess.run([result1])
        duration_2 = time.time() - start_time_2
        print "GPU", duration_2

