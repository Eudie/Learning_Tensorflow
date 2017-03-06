#!/usr/bin/python
# Author: Eudie

import pickle
import os
import sys
import re
import random
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import metrics
from ggplot import *
from sklearn.metrics import confusion_matrix


###############################################################
# Get tagged data from folder files
###############################################################


def to_dataframe(file_to_read, final_df):
    df_temp = pd.read_excel(file_to_read, skiprows=8, parse_cols="B:D")
    final_df = final_df.append(df_temp, ignore_index=True)
    return final_df


def getting_data():
    flag_name = sys.argv[1]
    df = pd.DataFrame(columns=['Link', 'Main Article', 'Category_1'])
    if os.path.splitext(flag_name)[1] == ".xlsx":
        df = to_dataframe(flag_name, df)
    else:
        list_of_files = os.listdir(flag_name)
        for individual_file in list_of_files:
            if os.path.splitext(individual_file)[1] == ".xlsx":
                df = to_dataframe(flag_name + "/" + individual_file, final_df=df)
    return df


input_as_dataframe = getting_data()
input_as_dataframe = input_as_dataframe.dropna(subset=['Category_1'])
input_as_dataframe = input_as_dataframe.dropna(subset=['Link'])

print input_as_dataframe['Category_1'].value_counts()
###############################################################
# Pulling prebuilt word2vec embedding
###############################################################

embedding_data = pickle.load(open('word2vec_model_4096_100000_full_data_1.pkl'))
final_embeddings = embedding_data['final_embeddings']
reverse_dictionary = embedding_data['reverse_dictionary']
dictionary = embedding_data['dictionary']

###############################################################
# Cleaning text
###############################################################

pattern = re.compile('[^a-z]', re.UNICODE)
articles = input_as_dataframe["Main Article"].tolist()
articles = map(lambda x: x.encode('utf-8', errors='ignore'), articles)
articles = map(lambda x: x.lower(), articles)
articles = map(lambda x: re.sub(pattern, ' ', x), articles)
articles = map(lambda x: x.split(), articles)

###############################################################
# Convert data into desire format to train
###############################################################

category = input_as_dataframe["Category_1"].tolist()
test_output = [[1, 0] if y == 'Fitness' else [0, 1] for y in category]


def convert2vec(embeddings_from_pickle, dictionary_from_pickle, input_sentences):
    lengths = []
    for i in range(len(input_sentences)):
        sentence = input_sentences[i]
        lengths.append(len(sentence))
        for j in range(len(sentence)):
            if sentence[j] not in dictionary:
                input_sentences[i][j] = embeddings_from_pickle[dictionary_from_pickle["UNK"]]
            else:
                input_sentences[i][j] = embeddings_from_pickle[dictionary_from_pickle[sentence[j]]]
    return input_sentences, lengths


test_input, test_length = convert2vec(final_embeddings, dictionary, articles)

###############################################################
# Divide data into test and train
###############################################################


batch_size = 16
lstm_size = 256
num_labels = 2
no_of_lstm_layers = 1

word2vec_sentences_idxes = range(len(test_output))

###############################################################
# Building tensorflow graph
###############################################################

graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    sentences = tf.placeholder(tf.float32, shape=(batch_size, None, 128))
    sequence_length = tf.placeholder(tf.float32, shape=batch_size)
    labels = tf.placeholder(tf.float32, shape=(batch_size, 2))

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
    stacked_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([stacked_lstm] * no_of_lstm_layers)
    outputs, states = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                        inputs=sentences,
                                        sequence_length=sequence_length,
                                        initial_state=stacked_lstm.zero_state(batch_size, tf.float32))
    # print(outputs.get_shape())
    output = tf.reduce_mean(outputs, [1])
    # val = tf.transpose(outputs, [1, 0, 2])
    # last = tf.gather(val, int(val.get_shape()[0]) - 1)
    # print(output.get_shape())
    # print(val.get_shape())
    # print(last.get_shape())
    softmax_w = tf.get_variable("softmax_w", [lstm_size, num_labels])
    softmax_b = tf.get_variable("softmax_b", [num_labels])
    prediction = tf.nn.softmax(tf.matmul(output, softmax_w) + softmax_b)

    cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(labels, 1), tf.argmax(prediction, 1))
    error = tf.reduce_sum(tf.cast(mistakes, tf.float32))

    saver = tf.train.Saver(tf.all_variables())
    init = tf.initialize_all_variables()

###############################################################
# Running the session
###############################################################


with tf.Session(graph=graph) as sess:
    sess = tf.Session(config=tf.ConfigProto())
    sess.run(init)
    saver.restore(sess, "saved_models/test_32_2_200.ckpt")

    ###############################################################
    # Prediction on test data
    ###############################################################
    link_in_output = []
    predicted = []
    main_article = []
    actual_category = []
    for test_buckets in xrange(len(test_input) / batch_size):
        temp_test_input = test_input[batch_size * test_buckets:batch_size * (test_buckets + 1)]
        test_max_len = max(test_length[batch_size * test_buckets:batch_size * (test_buckets + 1)])
        test_data = np.zeros((batch_size, test_max_len, 128), np.float32)
        for idx, t in enumerate(temp_test_input):
            t = np.array(t)

            test_data[idx, :t.shape[0], :] = t

        actual_category_batch = test_output[batch_size * test_buckets:batch_size * (test_buckets + 1)]
        test_labels = np.array(actual_category_batch, 'float32')

        predicted_batch = sess.run(prediction, feed_dict={sentences: test_data,
                                                          labels: test_labels,
                                                          sequence_length:
                                                              test_length[batch_size * test_buckets:batch_size * (
                                                                  test_buckets + 1)]})

        main_article_batch = input_as_dataframe['Main Article'][
                             batch_size * test_buckets:batch_size * (test_buckets + 1)]
        link_in_output_batch = input_as_dataframe['Link'][
                               batch_size * test_buckets:batch_size * (test_buckets + 1)]

        predicted.extend(predicted_batch[:, 0])
        actual_category.extend(test_labels[:, 0])
        main_article.extend(main_article_batch)
        link_in_output.extend(link_in_output_batch)


cutoff = 0.28
output_dataframe = pd.DataFrame({'Main Article': main_article, 'is_fitness': actual_category,
                                 'Prediction': predicted, 'Link': link_in_output})
output_dataframe['predict_class'] = np.where(output_dataframe['Prediction'] >= cutoff, 1, 0)
# print(output_dataframe)
# output_dataframe.to_csv("output_of_rnn_classifier_dec_to_july.csv",
#                         encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
# got optimal cutoff from chart = 0.25
fpr, tpr, threshold = metrics.roc_curve(actual_category, predicted)
auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=threshold))
print df.ix[df['threshold'] <= 0.281]
print confusion_matrix(output_dataframe['is_fitness'], output_dataframe['predict_class'])
print ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed') + \
      ggtitle("ROC Curve w/ AUC=%s" % str(auc))
print ggplot(output_dataframe, aes(x='Prediction', fill='is_fitness')) + geom_density() + facet_wrap("is_fitness")
