import re
import os
import time
import math
import shutil
import numpy as np
import tensorflow as tf
import logging
from collections import defaultdict
# import pickle
# import json
# import pandas as pd
# from collections import Counter
# import sys
# import itertools
# import zipfile
# import csv
# from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

LINE_FROM_CLASS = 5000
MINIMUM_ROW_LENGTH = 25
MAXIMUM_ROW_LENGTH = 150
BATCH_SIZE = 32
LSTM_HIDDEN_UNITS = 64
LSTM_TYPE = 'basic'
EPOCHS = 1
KEEP_PROB = 0.5
SHOULD_SAVE = True
# GENERAL VARS
PRO_FLD = '../'
DATA_DIR = 'input/'
EMB_FILE = 'glove.6B.50d.txt'
EMB_DIM = 50
EMB_FILE_PATH = PRO_FLD + DATA_DIR + EMB_FILE
# DATA_FILE = '2way_rus_usa{}-{}.txt'.format(MINIMUM_ROW_LENGTH, MAXIMUM_ROW_LENGTH)
DATA_FILE = '2way_short{}-{}'.format(MINIMUM_ROW_LENGTH, MAXIMUM_ROW_LENGTH)
DATA_FILE_PATH = PRO_FLD + DATA_DIR + DATA_FILE + '.txt'
COUNT_WORD = 20  # if a sentence has COUNT_WORD of the same word - it's a bad sentence (just a troll)

# existing FILES
MODEL_PATH = '../model_temp/model.ckpt'  # Should set it to model path if TRAIN = False
USE_TMP_FOLDER = True
TRAIN = False
TEST = True
PRINT_CLASSES_STATS_EACH_X_STEPS = 1  # prints dev stats each x steps


def clean_str(s):  # DATA
    strip_special_chars = re.compile("[^A-Za-z0-9 ,.]+")
    s = s.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", s)


# for a sentence: get each token index in word_to_emb_mat_ind (if doesn't exist take last index)
def convert_data_to_indices_of_emb_mat(sentence):
    words_ind = np.zeros(MAXIMUM_ROW_LENGTH, dtype='int32')
    for i in range(len(sentence)):
        token = sentence[i]
        if i >= MAXIMUM_ROW_LENGTH:  # fail safe - shouldn't happen
            break
        # if word_to_emb_mat_ind.contains(token) save index. else save index==len(emb_mat) - for all unseen words
        words_ind[i] = gl_word_to_emb_mat_ind.get(token, len(gl_word_to_emb_mat_ind))
    return words_ind


def convert_data_to_word_indices(data_x):
    data_x_emb_indices = []
    for sentence in data_x:
        data_x_emb_indices.append(convert_data_to_indices_of_emb_mat(sentence))
    return np.matrix(data_x_emb_indices)


def load_data(data_full_path, shuffle=False):
    all_lines, data_x, labels_str, labels_int = [], [], [], []
    with open(data_full_path, 'r', encoding="utf8") as data_file:
        for line in data_file:
            all_lines.append(clean_str(line))

    print('Total data size is {}'.format(len(all_lines)))
    if shuffle:
        np.random.shuffle(all_lines)  # will affect the train test split

    for line in all_lines:
        split_line = line.split()
        label = split_line[0:1]
        sentence = split_line[1:]
        labels_str.append(label[0])
        data_x.append(sentence)

    l_unique_labels_list = np.unique(np.array(labels_str))
    l_unique_labels_to_ind = {}
    l_unique_ind_to_labels = {}
    for i in range(len(l_unique_labels_list)):
        l_unique_labels_to_ind[l_unique_labels_list[i]] = i
        l_unique_ind_to_labels[i] = l_unique_labels_list[i]

    print('Our {} labels to index dictionary ={}'.format(len(l_unique_labels_to_ind), l_unique_labels_to_ind))
    print('Our {} index to labels dictionary ={}'.format(len(l_unique_ind_to_labels), l_unique_ind_to_labels))

    for i in range(len(labels_str)):
        labels_int.append(l_unique_labels_to_ind[labels_str[i]])

    # split data to train - test
    split_train_test_percent = 0.9
    split_ind = math.floor(len(labels_int) * split_train_test_percent)

    l_test_x = data_x[split_ind:]
    l_test_y = labels_int[split_ind:]

    l_train_dev_x = data_x[:split_ind]
    l_train_dev_y = labels_int[:split_ind]

    # split train data to train - dev
    split_train_dev_percent = 0.9
    split_ind2 = math.floor(len(l_train_dev_y) * split_train_dev_percent)

    l_dev_x = l_train_dev_x[split_ind2:]
    l_dev_y = l_train_dev_y[split_ind2:]

    l_train_x = l_train_dev_x[:split_ind2]
    l_train_y = l_train_dev_y[:split_ind2]

    # convert words to their index in the embedding matrix
    l_train_x = convert_data_to_word_indices(l_train_x)
    l_dev_x = convert_data_to_word_indices(l_dev_x)
    l_test_x = convert_data_to_word_indices(l_test_x)

    print('x_train: {}, x_dev: {}, x_test: {}'.format(len(l_train_x), len(l_dev_x), len(l_test_x)))
    print('y_train: {}, y_dev: {}, y_test: {}'.format(len(l_train_y), len(l_dev_y), len(l_test_y)))

    return l_train_x, l_train_y, l_dev_x, l_dev_y, l_test_x, l_test_y, l_unique_labels_to_ind, l_unique_ind_to_labels


# creates 2 objects
# l_word_to_emb_mat_ind: e.g. {'the' : 0, ',':1 ... }
#       the number of a key is the index in the l_emb_mat with leads to a EMB_DIM vector of floats
# l_emb_mat in the size len(l_word_to_emb_mat_ind) * EMB_DIM
def load_emb(emb_full_path):
    l_word_to_emb_mat_ind, l_emb_mat = {}, []
    with open(emb_full_path, 'r', encoding="utf8") as emb_file:
        for i, line in enumerate(emb_file.readlines()):
            split_line = line.split()
            l_word_to_emb_mat_ind[split_line[0]] = i
            embedding = np.array([float(val) for val in split_line[1:]], dtype='float32')
            l_emb_mat.append(embedding)

    # adding one more entry for all words that doesn't exist in the emb_full_path 
    l_emb_mat.append(np.zeros(EMB_DIM, dtype='float32'))
    print('Embedding tokens size={}'.format(len(l_emb_mat)))
    return l_word_to_emb_mat_ind, np.matrix(l_emb_mat, dtype='float32')


def get_bidirectional_rnn_model(l_emb_mat):
    tf.reset_default_graph()
    num_classes = len(gl_label_to_ind)
    input_data_x_batch = tf.placeholder(tf.int32, [BATCH_SIZE, MAXIMUM_ROW_LENGTH])
    input_labels_batch = tf.placeholder(tf.float32, [BATCH_SIZE, num_classes])
    keep_prob_pl = tf.placeholder(tf.float32)
    print("input_data_x_batch shape: {}".format(input_data_x_batch.get_shape()))
    print("input_labels_batch shape: {}".format(input_labels_batch.get_shape()))

    data = tf.nn.embedding_lookup(l_emb_mat, input_data_x_batch)

    if LSTM_TYPE == 'basic':
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=LSTM_HIDDEN_UNITS)
    else:
        lstm_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=LSTM_HIDDEN_UNITS)
    print("lstm_fw_cell units: {}".format(LSTM_HIDDEN_UNITS))
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=keep_prob_pl, dtype=tf.float32)

    if LSTM_TYPE == 'basic':
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=LSTM_HIDDEN_UNITS)
    else:
        lstm_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=LSTM_HIDDEN_UNITS)
    print("lstm_bw_cell units: {}".format(LSTM_HIDDEN_UNITS))
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=keep_prob_pl, dtype=tf.float32)

    outputs_as_vecs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, data, dtype=tf.float32)

    outputs_as_vecs = tf.concat(outputs_as_vecs, 2)
    outputs_as_vecs = tf.transpose(outputs_as_vecs, [1, 0, 2])

    weight = tf.Variable(tf.truncated_normal([2 * LSTM_HIDDEN_UNITS, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    outputs_as_value = tf.gather(outputs_as_vecs, int(outputs_as_vecs.get_shape()[0]) - 1)
    prediction = (tf.matmul(outputs_as_value, weight) + bias)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_labels_batch, 1))

    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    l_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=input_labels_batch))
    l_optimizer = tf.train.AdamOptimizer().minimize(l_loss)
    l_num_correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    return input_data_x_batch, input_labels_batch, keep_prob_pl, l_optimizer, l_loss, acc, l_num_correct, correct_pred


def convert_to_array(label_value):
    label_zero_one_vec = [0] * len(gl_label_to_ind)
    label_zero_one_vec[label_value - 1] = 1
    return label_zero_one_vec


def get_batch_sequential(data_x, data_y, batch_num, batch_size):
    batch = data_x[batch_num * batch_size:(batch_num + 1) * batch_size]
    labels = [convert_to_array(label) for label in data_y[batch_num * batch_size:(batch_num + 1) * batch_size]]

    return batch, labels


def train_step_func(sess, x_batch, y_batch):
    feed_dict = {input_data: x_batch,
                 input_labels: y_batch,
                 keep_prob: KEEP_PROB
                 }
    _, batch_loss_trn, batch_acc_trn = sess.run([optimizer, loss, accuracy], feed_dict)
    return batch_loss_trn, batch_acc_trn


def dev_step_func(sess, x_batch, y_batch):
    feed_dict = {input_data: x_batch,
                 input_labels: y_batch,
                 keep_prob: 1.0}
    batch_loss_dev, batch_acc_dev, batch_num_correct, batch_predictions = sess.run([loss, accuracy, num_correct, predictions], feed_dict)
    # loss_l, accuracy_l, num_correct, predictions_l = sess.run(
    #     [cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
    # return accuracy_l, loss_l, num_correct, predictions_l
    return batch_loss_dev, batch_acc_dev, batch_num_correct, batch_predictions


def print_stats(stat_dict_total, stat_dict_correct):
    longest_key = 0
    for key in stat_dict_total:
        if len(key) > longest_key:
            longest_key = len(key)
    for key in stat_dict_total:
        my_msg = "        Class {:{}s}: ({}/{}) -> accuracy: {:.4f}%"
        temp = 0
        if key in stat_dict_correct:
            temp = stat_dict_correct[key]
        my_acc_l = (float(temp) / float(stat_dict_total[key])) * 100
        print(my_msg.format(key, longest_key, temp, stat_dict_total[key], my_acc_l))
    return


def train(l_train_x, l_train_y, l_dev_x, l_dev_y):
    timestamp = str(int(time.time()))
    if USE_TMP_FOLDER:
        timestamp = "temp"
    model_dir = PRO_FLD + 'model_' + timestamp + '/'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    model_full_path = '{}model.ckpt'.format(model_dir)

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        best_accuracy, best_at_epoch = 0, 0
        batches_num_train = int(math.ceil(len(l_train_y) / BATCH_SIZE))
        batches_num_dev = int(math.ceil(len(l_dev_y) / BATCH_SIZE))
        for i in range(EPOCHS):
            print("Epoch: {0}/{1}".format((i + 1), EPOCHS))
            for train_step in range(batches_num_train):
                batch_x_trn, batch_y_trn = get_batch_sequential(l_train_x, l_train_y, train_step, BATCH_SIZE)
                if len(batch_y_trn) != BATCH_SIZE:
                    continue
                _, _ = train_step_func(sess, batch_x_trn, batch_y_trn)
                # #if you like to print the train data performance, replace the above line with the next 3 lines
                # batch_loss_trn, batch_acc_trn = train_step_func(sess, batch_x_trn, batch_y_trn)
                # msg = "    TRAIN: STEP {}/{}: batch_acc = {:.4f}% , batch loss = {:.4f}"
                # print(msg.format(train_step + 1, batches_num_train - 1, batch_acc_trn * 100, batch_loss_trn))

                # check on dev data
                total_correct, total_seen, dev_acc = 0, 0, 0
                stat_dict_step_total, stat_dict_step_correct = defaultdict(int), defaultdict(int)
                for dev_step in range(batches_num_dev):
                    batch_x_dev, batch_y_dev = get_batch_sequential(l_dev_x, l_dev_y, dev_step, BATCH_SIZE)
                    if len(batch_y_dev) != BATCH_SIZE:
                        continue
                    _, _, batch_num_correct_dev, l_predictions = dev_step_func(sess, batch_x_dev, batch_y_dev)
                    # #if you like to print the dev data performance, replace the above line with the next 3 lines
                    # batch_loss_dev, batch_acc_dev, batch_num_correct_dev, l_predictions = dev_step_func(sess, batch_x_dev, batch_y_dev)
                    # msg = "        DEV: STEP {}/{}: batch_acc = {:.4f}% , batch loss = {:.4f}"
                    # print(msg.format(dev_step + 1, batches_num_dev - 1, batch_acc_dev * 100, batch_loss_dev))

                    for p in range(len(l_predictions)):  # calculating acc per class
                        true_val = int(np.argmax(batch_y_dev[p]))
                        true_lbl = gl_ind_to_label[true_val]
                        stat_dict_step_total[true_lbl] += 1
                        if l_predictions[p]:
                            stat_dict_step_correct[true_lbl] += 1

                    total_correct += int(batch_num_correct_dev)  # sum correct predictions for acc
                    total_seen += BATCH_SIZE

                dev_acc = total_correct/total_seen
                msg = "    DEV accuracy on epoch {}/{} = {:.4f}%"
                print(msg.format(i + 1, EPOCHS, dev_acc * 100))
                print_stats(stat_dict_step_total, stat_dict_step_correct)
                if dev_acc > best_accuracy:
                    best_accuracy, best_at_epoch = dev_acc, i + 1
                    if SHOULD_SAVE:
                        save_path = saver.save(sess, model_full_path)
                        logging.info('    Saved model {} at epoch {}'.format(save_path, best_at_epoch))
                    msg = '    Best accuracy {:.4f}% at epoch {}/{} ({}/{})'
                    logging.info(msg.format(best_accuracy * 100, best_at_epoch, EPOCHS, total_correct, total_seen))
            print("###################################################################################################")
            train_msg = '***Training is complete. Best accuracy {:.4f}% at step {}/{}'
            print(train_msg.format(best_accuracy * 100, best_at_epoch, EPOCHS))
    return model_full_path, best_accuracy


def test(l_model_full_path, l_test_x, l_test_y):
    print('***Testing...')
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, l_model_full_path)

        total_correct, total_seen, l_test_acc = 0, 0, 0
        test_stat_dict_total, test_dict_correct = defaultdict(int), defaultdict(int)
        batches_num_test = int(math.ceil(len(l_test_y) / BATCH_SIZE))
        for test_step in range(batches_num_test):
            x_test_batch, y_test_batch = get_batch_sequential(l_test_x, l_test_y, test_step, BATCH_SIZE)
            if len(y_test_batch) != BATCH_SIZE:
                continue
            _, _, batch_num_correct_test, l_predictions = dev_step_func(sess, x_test_batch, y_test_batch)
            for p in range(len(l_predictions)):  # calculating acc per class
                true_val = int(np.argmax(y_test_batch[p]))
                true_lbl = gl_ind_to_label[true_val]
                test_stat_dict_total[true_lbl] += 1
                if l_predictions[p]:
                    test_dict_correct[true_lbl] += 1

            total_correct += int(batch_num_correct_test)  # sum correct predictions for acc
            total_seen += BATCH_SIZE

        l_test_acc = total_correct/total_seen
        acc_msg = '    Accuracy on test set - ({}/{}) -> accuracy: {:.4f}%'
        print(acc_msg.format(total_correct, total_seen, l_test_acc*100))
        print_stats(test_stat_dict_total, test_dict_correct)  # Stats prints
    return l_test_acc


def args_print(stage, mdl_path, l_data_size, l_trn_acc, l_test_acc, duration=0):
    print("{} ----------------------".format(stage))
    print("data:")
    print("     DATA_FILE_PATH is {}".format(DATA_FILE_PATH))
    print("     MINIMUM_ROW_LENGTH is {}".format(MINIMUM_ROW_LENGTH))
    print("     MAXIMUM_ROW_LENGTH is {}".format(MAXIMUM_ROW_LENGTH))
    print("     COUNT_WORD is {}".format(COUNT_WORD))
    print("     LINE_FROM_CLASS is {}".format(LINE_FROM_CLASS))
    print("     Total data size is {}".format(l_data_size))

    print("embedding:")
    print("     EMB_FILE_PATH {}".format(EMB_FILE_PATH))
    print("     EMB_DIM {}".format(EMB_DIM))
    print("     EMB_WORDS_COUNT {}".format(len(gl_word_to_emb_mat_ind)+1))

    print("run config:")
    print("     EPOCHS {}".format(EPOCHS))
    print("     BATCH_SIZE {}".format(BATCH_SIZE))
    print("     KEEP_PROB {}".format(KEEP_PROB))
    print("     BATCH_SIZE {}".format(BATCH_SIZE))
    print("     LSTM_HIDDEN_UNITS {}".format(LSTM_HIDDEN_UNITS))
    print("     LSTM_CELL_TYPE {}".format(LSTM_TYPE))

    print("model:")
    print("     USE_TMP_FOLDER {}".format(USE_TMP_FOLDER))
    print("     mdl_path {}".format(mdl_path))

    print("results:")
    print("     best training acc {}".format(l_trn_acc * 100))
    print("     testing acc {}".format(l_test_acc * 100))

    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time(HH:MM:SS): {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))
    return


if __name__ == '__main__':
    print("Entering function __main__")
    total_start_time, trn_acc, test_acc = time.time(), 0, 0
    global gl_word_to_emb_mat_ind, gl_label_to_ind, gl_ind_to_label
    gl_word_to_emb_mat_ind, emb_mat = load_emb(EMB_FILE_PATH)
    train_x, train_y, dev_x, dev_y, test_x, test_y, gl_label_to_ind, gl_ind_to_label = load_data(DATA_FILE_PATH)
    input_data, input_labels, keep_prob, optimizer, loss, accuracy, num_correct, predictions = get_bidirectional_rnn_model(emb_mat)
    if TRAIN:
        MODEL_PATH, trn_acc = train(train_x, train_y, dev_x, dev_y)
    if TEST:
        test_acc = test(MODEL_PATH, test_x, test_y)
    dur = time.time() - total_start_time
    data_size = len(train_y) + len(dev_y) + len(test_y)
    args_print('End summary', MODEL_PATH, data_size, trn_acc, test_acc, int(dur))
    print("Leaving function __main__")
