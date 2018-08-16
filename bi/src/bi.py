import re
import os
import time
import math
import shutil
import numpy as np
import tensorflow as tf
import logging
from collections import defaultdict
import io
import sys

logging.getLogger().setLevel(logging.INFO)

MINIMUM_ROW_LENGTH = 25
MAXIMUM_ROW_LENGTH = 150
LSTM_HIDDEN_UNITS = 64
LSTM_TYPE = 'GRU'
EPOCHS = 20
BATCH_SIZE = 100
KEEP_PROB = 0.5
SHOULD_SAVE = True

PRO_FLD = '../'
DATA_DIR = 'input/'
EMB_FILE = 'glove.6B.50d.txt'
EMB_DIM = 50
EMB_FILE_PATH = PRO_FLD + DATA_DIR + EMB_FILE
# DATA_FILE = '2way_rus_usa_v2_{}-{}'.format(MINIMUM_ROW_LENGTH, MAXIMUM_ROW_LENGTH)
# DATA_FILE = '4way_tur_ger_rus_usa{}-{}'.format(MINIMUM_ROW_LENGTH, MAXIMUM_ROW_LENGTH)
DATA_FILE = '5way_tur_ger_rus_fra_usa{}-{}'.format(MINIMUM_ROW_LENGTH, MAXIMUM_ROW_LENGTH)
DATA_FILE_PATH = PRO_FLD + DATA_DIR + DATA_FILE + '.txt'
COUNT_WORD = 20  # if a sentence has COUNT_WORD of the same word - it's a bad sentence (just a troll)

UNIT_TESTING = False  # will print test results

MODEL_PATH = '../model_temp/model.ckpt'  # Should set it to model path if TRAIN = False
USE_TMP_FOLDER = True
TRAIN = True
TEST = True

# uncomment for local run
# DATA_FILE = '2way_short{}-{}'.format(MINIMUM_ROW_LENGTH, MAXIMUM_ROW_LENGTH)
# DATA_FILE = '2way_duplicated_data_rus_usa25-150{}-{}'.format(MINIMUM_ROW_LENGTH, MAXIMUM_ROW_LENGTH)
# DATA_FILE_PATH = PRO_FLD + DATA_DIR + DATA_FILE + '.txt'
# EPOCHS = 3
# BATCH_SIZE = 10
# TEST = False
# SHOULD_SAVE = False


def clean_str(s):  # removing all chars but letters, numbers, spaces, commas and dots
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


# piles the new representation of the sentences in a matrix (@see  convert_data_to_indices_of_emb_mat(sentence))
def convert_data_to_word_indices(data_x):
    data_x_emb_indices = []
    for sentence in data_x:
        data_x_emb_indices.append(convert_data_to_indices_of_emb_mat(sentence))
    return np.matrix(data_x_emb_indices)


# loads and prepare data (creates train,dev,test datum. also 2 dicts to help convert class value to class label)
# data_x will return in the shape of indices in the embedding matrix.
# data_t will return in the shape of numeric value of the class stored in the 2 dicts
def load_data(data_full_path, shuffle=False):
    all_lines, data_x, labels_str, labels_int = [], [], [], []
    with io.open(data_full_path, 'r', encoding="utf-8") as data_file:
        for line in data_file:
            all_lines.append(clean_str(line))

    print('File name {}. Total data size is {}'.format(DATA_FILE, len(all_lines)))
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

    line_from_class = len(all_lines)/len(l_unique_labels_to_ind)
    print('Our {} labels to index dictionary ={}'.format(len(l_unique_labels_to_ind), l_unique_labels_to_ind))
    print('Our {} index to labels dictionary ={}'.format(len(l_unique_ind_to_labels), l_unique_ind_to_labels))

    if UNIT_TESTING:
        print('\t\t\t\t\t{} LOAD_DATA({}): len(all_lines)==10K?'.format(len(all_lines) == 10000, DATA_FILE))
        print('\t\t\t\t\t{} LOAD_DATA({}): all_lines[0] start with \'us i think if you\'?'.format(all_lines[0].startswith('us i think if you'), DATA_FILE))
        print('\t\t\t\t\t{} LOAD_DATA({}): all_lines[-1] start with \'us if slovenia did nt\'?'.format(all_lines[-1].startswith('us if slovenia did nt'), DATA_FILE))  # 2 way us rus

    for i in range(len(labels_str)):
        labels_int.append(l_unique_labels_to_ind[labels_str[i]])

    # split data to train - test
    split_train_test_percent = 0.9
    split_ind = int(math.floor(len(labels_int) * split_train_test_percent))

    l_test_x = data_x[split_ind:]
    l_test_y = labels_int[split_ind:]

    l_train_dev_x = data_x[:split_ind]
    l_train_dev_y = labels_int[:split_ind]

    # split train data to train - dev
    split_train_dev_percent = 0.9
    split_ind2 = int(math.floor(len(l_train_dev_y) * split_train_dev_percent))

    l_dev_x = l_train_dev_x[split_ind2:]
    l_dev_y = l_train_dev_y[split_ind2:]

    l_train_x = l_train_dev_x[:split_ind2]
    l_train_y = l_train_dev_y[:split_ind2]

    if UNIT_TESTING:
        print('\t\t\t\t\t{} LOAD_DATA({}): l_train_x[0] which is line {} start with \'i\'?'.format(l_train_x[0][0] == 'i', DATA_FILE, 1))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_train_y[0] which is line {} is us?'.format(l_unique_ind_to_labels[l_train_y[0]] == 'us', DATA_FILE, 1))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_train_x[-1] which is line {} start with \'not\'?'.format(l_train_x[-1][0] == 'not', DATA_FILE, split_ind2))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_train_y[-1] which is line {} is us?'.format(l_unique_ind_to_labels[l_train_y[-1]] == 'us', DATA_FILE, split_ind2))

        print('\t\t\t\t\t{} LOAD_DATA({}): l_dev_x[0] which is line {} start with \'the\'?'.format(l_dev_x[0][0] == 'the', DATA_FILE, split_ind2+1))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_dev_y[0] which is line {} is us?'.format(l_unique_ind_to_labels[l_dev_y[0]] == 'us', DATA_FILE, split_ind2+1))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_dev_x[-1] which is line {} start with \'the\'?'.format(l_dev_x[-1][0] == 'the', DATA_FILE, len(l_train_dev_y)))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_dev_y[-1] which is line {} is russia?'.format(l_unique_ind_to_labels[l_dev_y[-1]] == 'russia', DATA_FILE, len(l_train_dev_y)))

        print('\t\t\t\t\t{} LOAD_DATA({}): l_test_x[0] which is line {} start with \'but\'?'.format(l_test_x[0][0] == 'but', DATA_FILE, split_ind+1))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_test_y[0] which is line {} is us?'.format(l_unique_ind_to_labels[l_test_y[0]] == 'us', DATA_FILE, split_ind+1))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_test_x[-1] which is line {} start with \'if\'?'.format(l_test_x[-1][0] == 'if', DATA_FILE, len(all_lines)))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_test_y[-1] which is line {} is us?'.format(l_unique_ind_to_labels[l_test_y[-1]] == 'us', DATA_FILE, len(all_lines)))

    l_train_x = convert_data_to_word_indices(l_train_x)
    l_dev_x = convert_data_to_word_indices(l_dev_x)
    l_test_x = convert_data_to_word_indices(l_test_x)
    if UNIT_TESTING:
        print('\t\t\t\t\t{} LOAD_DATA({}): l_train_x[0].item(0, 0) (which is \'i\') == 41(index in gl_word_to_emb_mat_ind)?'.format(l_train_x[0].item(0, 0) == gl_word_to_emb_mat_ind['i'], DATA_FILE))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_dev_x[0].item(0, 0) (which is \'the\') == 41(index in gl_word_to_emb_mat_ind)?'.format(l_dev_x[0].item(0, 0) == gl_word_to_emb_mat_ind['the'], DATA_FILE))
        print('\t\t\t\t\t{} LOAD_DATA({}): l_test_x[0].item(0, 0) (which is \'but\') == 41(index in gl_word_to_emb_mat_ind)?'.format(l_test_x[0].item(0, 0) == gl_word_to_emb_mat_ind['but'], DATA_FILE))

    print('x_train: {}, x_dev: {}, x_test: {}'.format(len(l_train_x), len(l_dev_x), len(l_test_x)))
    print('y_train: {}, y_dev: {}, y_test: {}'.format(len(l_train_y), len(l_dev_y), len(l_test_y)))

    return l_train_x, l_train_y, l_dev_x, l_dev_y, l_test_x, l_test_y, l_unique_labels_to_ind, l_unique_ind_to_labels, line_from_class


# creates 2 objects
# l_word_to_emb_mat_ind: e.g. {'the' : 0, ',':1 ... }
#       the number of a key is the index in the l_emb_mat with leads to a EMB_DIM vector of floats
# l_emb_mat in the size len(l_word_to_emb_mat_ind) * EMB_DIM
def load_emb(emb_full_path):
    l_word_to_emb_mat_ind, l_emb_mat = {}, []
    with io.open(emb_full_path, 'r', encoding="utf8") as emb_file:
        for i, line in enumerate(emb_file.readlines()):
            split_line = line.split()
            l_word_to_emb_mat_ind[split_line[0]] = i
            embedding = np.array([float(val) for val in split_line[1:]], dtype='float32')
            l_emb_mat.append(embedding)

    # adding one more entry for all words that doesn't exist in the emb_full_path
    l_emb_mat.append(np.zeros(EMB_DIM, dtype='float32'))
    print('Embedding tokens size={}'.format(len(l_emb_mat)))
    if UNIT_TESTING:
        print('\t\t\t\t\t{} EMB: len(l_word_to_emb_mat_ind)==400K?'.format(len(l_word_to_emb_mat_ind) == 400000))
        print('\t\t\t\t\t{} EMB: l_word_to_emb_mat_ind[\'people\']==69?'.format(l_word_to_emb_mat_ind['people'] == 69))
        print('\t\t\t\t\t{} EMB: len(l_emb_mat)==400001?'.format(l_word_to_emb_mat_ind['people'] == 69))
        a = str(l_emb_mat[l_word_to_emb_mat_ind['people']][0])
        print('\t\t\t\t\t{} EMB: l_emb_mat[l_word_to_emb_mat_ind[\'people\']] start with 0.95281?'.format(a == '0.95281'))
    return l_word_to_emb_mat_ind, np.matrix(l_emb_mat, dtype='float32')


# bidirectional model creation
def get_bidirectional_rnn_model(l_emb_mat):
    tf.reset_default_graph()
    num_classes = len(gl_label_to_ind)
    input_data_x_batch = tf.placeholder(tf.int32, [BATCH_SIZE, MAXIMUM_ROW_LENGTH])
    input_labels_batch = tf.placeholder(tf.float32, [BATCH_SIZE, num_classes])
    keep_prob_pl = tf.placeholder(tf.float32)
    print("input_data_x_batch shape: {}".format(input_data_x_batch.get_shape()))
    print("input_labels_batch shape: {}".format(input_labels_batch.get_shape()))

    data = tf.nn.embedding_lookup(l_emb_mat, input_data_x_batch)

    gru_forward_cell = tf.contrib.rnn.GRUCell(num_units=LSTM_HIDDEN_UNITS)
    print("gru_forward_cell units: {}".format(LSTM_HIDDEN_UNITS))
    gru_forward_cell = tf.contrib.rnn.DropoutWrapper(cell=gru_forward_cell, output_keep_prob=keep_prob_pl, dtype=tf.float32)

    gru_backward_cell = tf.contrib.rnn.GRUCell(num_units=LSTM_HIDDEN_UNITS)
    print("gru_backward_cell units: {}".format(LSTM_HIDDEN_UNITS))
    gru_backward_cell = tf.contrib.rnn.DropoutWrapper(cell=gru_backward_cell, output_keep_prob=keep_prob_pl, dtype=tf.float32)

    outputs_as_vecs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_forward_cell, cell_bw=gru_backward_cell, inputs=data, dtype=tf.float32)

    outputs_as_vecs = tf.concat(outputs_as_vecs, 2)
    outputs_as_vecs = tf.transpose(outputs_as_vecs, [1, 0, 2])

    weight = tf.Variable(tf.truncated_normal([2 * LSTM_HIDDEN_UNITS, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    outputs_as_value = tf.gather(outputs_as_vecs, int(outputs_as_vecs.get_shape()[0]) - 1)
    prediction = (tf.matmul(outputs_as_value, weight) + bias)

    l_correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_labels_batch, 1))
    l_num_correct = tf.reduce_sum(tf.cast(l_correct_pred, tf.float32))

    acc = tf.reduce_mean(tf.cast(l_correct_pred, tf.float32))

    l_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=input_labels_batch))

    l_global_step = tf.Variable(0, name='global_step', trainable=False)
    # l_loss = tf.Print(l_loss, [l_loss, outputs_as_vecs, outputs_as_value])
    l_optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)

    grads_and_vars = l_optimizer.compute_gradients(l_loss)
    l_train_op = l_optimizer.apply_gradients(grads_and_vars, global_step=l_global_step)
    return input_data_x_batch, input_labels_batch, keep_prob_pl, l_train_op, l_global_step, l_loss, acc, l_num_correct, l_correct_pred


# e.g. 5 classes. takes the value 3 and returns [0 0 0 1 0]
def convert_to_array(label_value):
    label_zero_one_vec = [0] * len(gl_label_to_ind)
    label_zero_one_vec[label_value] = 1
    return label_zero_one_vec


# gets a portion of the data in the batch_num index
def get_batch_sequential(data_x, data_y, batch_num, batch_size):
    batch = data_x[batch_num * batch_size: (batch_num + 1) * batch_size]
    labels = [convert_to_array(label) for label in data_y[batch_num * batch_size: (batch_num + 1) * batch_size]]
    return batch, labels


# happen each train step - EFFECTS WEIGHTS !
def train_step_func(sess, x_batch, y_batch):
    feed_dict = {input_data: x_batch,
                 input_labels: y_batch,
                 keep_prob: KEEP_PROB
                 }
    _, gs, batch_loss_trn, batch_acc_trn = sess.run([train_op, global_step, loss, accuracy], feed_dict)
    return batch_loss_trn, batch_acc_trn


# happen each dev step or test step - DOESN'T effect weights !
def dev_step_func(sess, x_batch, y_batch):
    feed_dict = {input_data: x_batch,
                 input_labels: y_batch,
                 keep_prob: 1.0
                 }
    batch_loss_dev, batch_acc_dev, batch_num_correct, batch_predictions = sess.run([loss, accuracy, num_correct, correct_pred], feed_dict)
    return batch_loss_dev, batch_acc_dev, batch_num_correct, batch_predictions


# allows us to analyze the accuracy by printing the recall of each class
# prints all classes in turn with their accuracy
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


# trains the model on train data and evaluates the model 3 times per epoch on dev data
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
            print("Epoch: {}/{} ---- best so far on epoch {}: acc={:.4f}%".format((i + 1), EPOCHS, best_at_epoch, best_accuracy*100))
            for train_step in range(batches_num_train):
                batch_x_trn, batch_y_trn = get_batch_sequential(l_train_x, l_train_y, train_step, BATCH_SIZE)
                if len(batch_y_trn) != BATCH_SIZE:
                    print('len(batch_y_trn) != BATCH_SIZE - {}'.format(len(batch_y_trn)))
                    continue
                _, _ = train_step_func(sess, batch_x_trn, batch_y_trn)
                # #if you like to print the train data performance, replace the above line with the next 3 lines
                # batch_loss_trn, batch_acc_trn = train_step_func(sess, batch_x_trn, batch_y_trn)
                # msg = "    TRAIN: STEP {}/{}: batch_acc = {:.4f}% , batch loss = {:.4f}"
                # print(msg.format(train_step + 1, batches_num_train - 1, batch_acc_trn * 100, batch_loss_trn))

                # check on dev data 3 times per epoch
                if train_step == 0 or train_step == int(batches_num_train/2) or (train_step == batches_num_train - 1):
                    total_correct, total_seen, dev_acc = 0, 0, 0
                    stat_dict_step_total, stat_dict_step_correct = defaultdict(int), defaultdict(int)
                    for dev_step in range(batches_num_dev):
                        batch_x_dev, batch_y_dev = get_batch_sequential(l_dev_x, l_dev_y, dev_step, BATCH_SIZE)
                        if len(batch_y_dev) != BATCH_SIZE:
                            print('len(batch_y_dev) != BATCH_SIZE - {}'.format(len(batch_y_dev)))
                            continue
                        _, _, batch_num_correct_dev, l_predictions = dev_step_func(sess, batch_x_dev, batch_y_dev)
                        # #if you like to print the dev data performance, replace the above line with the next 3 lines
                        # batch_loss_dev, batch_acc_dev, batch_num_correct_dev, l_predictions = dev_step_func(sess, batch_x_dev, batch_y_dev)
                        # msg = "        DEV: STEP {}/{}: batch_acc = {:.4f}% , batch loss = {:.4f}"
                        # print(msg.format(dev_step + 1, batches_num_dev - 1, batch_acc_dev * 100, batch_loss_dev))

                        # print(batch_num_correct_dev, BATCH_SIZE)
                        # print(l_predictions)
                        # print(batch_y_dev)
                        for p in range(len(l_predictions)):  # calculating acc per class
                            true_val = int(np.argmax(batch_y_dev[p]))
                            true_lbl = gl_ind_to_label[true_val]
                            stat_dict_step_total[true_lbl] += 1
                            if bool(l_predictions[p]) is True:
                                stat_dict_step_correct[true_lbl] += 1

                        total_correct += int(batch_num_correct_dev)  # sum correct predictions for acc
                        total_seen += BATCH_SIZE

                    dev_acc = float(total_correct)/float(total_seen)
                    msg = "    DEV accuracy on epoch {}/{} in train step {} = {:.4f}%"
                    print(msg.format(i + 1, EPOCHS, train_step, dev_acc * 100))
                    print_stats(stat_dict_step_total, stat_dict_step_correct)
                    if dev_acc > best_accuracy:
                        best_accuracy, best_at_epoch = dev_acc, i + 1
                        if SHOULD_SAVE:
                            save_path = saver.save(sess, model_full_path)
                            logging.info('    Saved model {} at epoch {}'.format(save_path, best_at_epoch))
                        msg = '    Best accuracy {:.4f}% at epoch {}/{} ({}/{})'
                        logging.info(msg.format(best_accuracy * 100, best_at_epoch, EPOCHS, total_correct, total_seen))
            print("###################################################################################################")
        train_msg = '***Training is complete. Best accuracy {:.4f}% at epoch {}/{}'
        print(train_msg.format(best_accuracy * 100, best_at_epoch, EPOCHS))
    return model_full_path, best_accuracy


# restores the model from saved folder and test it on test data
# notice if you run test only, you need to set the MODEL_PATH to the model checkpoint file
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

        l_test_acc = float(total_correct) / float(total_seen)
        acc_msg = '    Accuracy on test set - ({}/{}) -> accuracy: {:.4f}%'
        print(acc_msg.format(total_correct, total_seen, l_test_acc*100))
        print_stats(test_stat_dict_total, test_dict_correct)  # Stats prints
    return l_test_acc


# print summary of the run
def args_print(stage, mdl_path, l_data_size, l_trn_acc, l_test_acc, l_lines_per_class, duration=0):
    print("{} ----------------------".format(stage))
    print("data:")
    print("     DATA_FILE_PATH is {}".format(DATA_FILE_PATH))
    print("     MINIMUM_ROW_LENGTH is {}".format(MINIMUM_ROW_LENGTH))
    print("     MAXIMUM_ROW_LENGTH is {}".format(MAXIMUM_ROW_LENGTH))
    print("     COUNT_WORD is {}".format(COUNT_WORD))
    print("     lines_per_class is {}".format(l_lines_per_class))
    print("     number of classes is {}".format(len(gl_label_to_ind)))
    print("     Total data size is {}".format(l_data_size))

    print("embedding:")
    print("     EMB_FILE_PATH {}".format(EMB_FILE_PATH))
    print("     EMB_DIM {}".format(EMB_DIM))
    print("     EMB_WORDS_COUNT {}".format(len(gl_word_to_emb_mat_ind)+1))

    print("run config:")
    print("     EPOCHS {}".format(EPOCHS))
    print("     evaluating on dev data 3 times per epoch")
    print("     KEEP_PROB {}".format(KEEP_PROB))
    print("     BATCH_SIZE {}".format(BATCH_SIZE))
    print("     LSTM_HIDDEN_UNITS {}".format(LSTM_HIDDEN_UNITS))
    print("     LSTM_CELL_TYPE {}".format(LSTM_TYPE))
    print("     optimizer is adamOptimizer - learn rate:  0.001")

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
    train_x, train_y, dev_x, dev_y, test_x, test_y, gl_label_to_ind, gl_ind_to_label, lines_per_class = load_data(DATA_FILE_PATH)
    input_data, input_labels, keep_prob, train_op, global_step, loss, accuracy, num_correct, correct_pred = get_bidirectional_rnn_model(emb_mat)
    if TRAIN:
        MODEL_PATH, trn_acc = train(train_x, train_y, dev_x, dev_y)
    if TEST:
        test_acc = test(MODEL_PATH, test_x, test_y)
    dur = time.time() - total_start_time
    data_size = len(train_y) + len(dev_y) + len(test_y)
    args_print('End summary', MODEL_PATH, data_size, trn_acc, test_acc, lines_per_class, int(dur))
    print("Leaving function __main__")
