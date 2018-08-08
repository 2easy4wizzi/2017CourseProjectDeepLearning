import re
# import pickle
# import json
import sys
import itertools
import zipfile
import os
import time
import shutil
import pandas as pd
from collections import Counter
import numpy as np
import tensorflow as tf
import logging
import csv
from sklearn.model_selection import train_test_split
from collections import defaultdict
from text_cnn_rnn import TextCNNRNN

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel(logging.INFO)

# GENERAL VARS
DATA_DIR = 'data/'
PRO_FLD = ''

# casting txt to csv.zip
# BASE_REGULAR = 'isr_france_hungary_poland_45t'
# BASE_REGULAR = 'us_vs_sp_45tTEMP'
# BASE_REGULAR = 'shortdata'
# BASE_REGULAR = 'us_vs_gerAndHol_45t'
BASE_REGULAR = 'us_vs_gerAndHol_25t'
# BASE_REGULAR = 'isr_france_hungary_45t'
REGULAR_FILE_TO_CSV = PRO_FLD + DATA_DIR + BASE_REGULAR + '.txt'
CSV_NAME = BASE_REGULAR + '.csv'
CSV_FULL_PATH = PRO_FLD + DATA_DIR + CSV_NAME
MINIMUM_ROW_LENGTH = 45
MAXIMUM_ROW_LENGTH = 150
COUNT_WORD = 20  # if a sentence has COUNT_WORD of the same word - it's a bad sentence (just a troll)
RAW_DATA_PATH = "rawData/"  # if RAW_DATA_PATH isn't in your project home dir, set this to the path of rawData folder
REDDIT_DIR = "reddit/"
NON_NATIVE_RAW_FOLDER_NAME = "non-native/"
NATIVE_RAW_FOLDER_NAME = "native/"

# existing FILES
TRA_FLD = 'trained_results_1533109035/'  # NOT used if USE_TMP_FOLDER is TRUE !!!
TRAIN_FILE_PATH = CSV_FULL_PATH + '.zip'

USE_TMP_FOLDER = True
MAKE_NEW_DATA_FILE = False  # if this is True, main will not run. a new CSV.ZIP will be created. change the params
IS_TRAIN = True
SHOULD_SAVE = True
RUN_TEST_AFTER_TRAIN = True and SHOULD_SAVE  # if SHOULD_SAVE is false can't restore and run test
PRINT_CLASSES_STATS_EACH_X_STEPS = 1  # prints dev stats each x steps
PRINT_WORD_PARAGRAPH = True

params = {'batch_size': 128,
          'dropout_keep_prob': 0.5,
          'embedding_dim': 300,
          'evaluate_every': 1,
          'filter_sizes': "3,4,5",
          'hidden_unit': 300,
          'l2_reg_lambda': 0.0,
          'max_pool_size': 4,
          'non_static': False,
          'num_epochs': 10,
          'num_filters': 32}


def clean_str(s):  # DATA
    strip_special_chars = re.compile("[^A-Za-z0-9 ,.]+")
    s = s.lower().replace("<br />", " ")
    # return s.lower()
    return re.sub(strip_special_chars, "", s)


def load_embeddings(vocabulary):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """Pad sentences during training or prediction"""
    # longest_sen = ""
    # max_found = 0
    # for sen in sentences:
    #     if len(sen) > max_found:
    #         longest_sen = sen
    #         max_found = len(sen)
    # print(max_found)
    # print(longest_sen)

    if forced_sequence_length is None:  # Train
        sequence_length = max(len(x) for x in sentences)
    else:  # Prediction
        print('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    m_len = 'All sentences length (after padding) will be {} words(length in words of the longest sentence)'
    print(m_len.format(sequence_length))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:  # Prediction: cut off the sentence if it is longer than the sequence length
            print('This sentence has to be cut off because it is longer than trained sequence length')
            exit(0)
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    shuffle = False  # TODO remove line
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
    return


def load_data(filename):
    print("Entering function load_data - data file name is {}".format(filename))
    df = pd.read_csv(filename, compression='zip')
    selected = ['Category', 'Descript']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    # df = df.reindex(np.random.permutation(df.index))  # TODO remove comment

    labels = sorted(list(set(df[selected[0]].tolist())))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw = df[selected[1]].apply(lambda x_l: clean_str(x_l).split(' ')).tolist()
    y_raw = df[selected[0]].apply(lambda y_l: label_dict[y_l]).tolist()
    # print(len(x_raw[0]))
    # print(x_raw[0])
    for i in range(len(x_raw)):  # remove empty strings
        x_raw[i] = [x for x in x_raw[i] if x]
    # print(len(x_raw[0]))
    # print(x_raw[0])

    # REMOVE TOP 10 SENTENCES? maybe we should
    # len_sen = []
    # print(len(x_raw))
    # for s in x_raw:
    #     len_sen.append(len(s))
    # len_sen = sorted(len_sen, reverse=True)[:10]
    # ind_to_remove = []
    # for i in range(len(x_raw)):
    #     if len(x_raw[i]) in len_sen:
    #         ind_to_remove.append(i)
    #
    # ind_to_remove = sorted(ind_to_remove, reverse=True)
    # for i in range(len(ind_to_remove)):
    #     a = ind_to_remove[i]
    #     del y_raw[a]
    #     del x_raw[a]
    # print(len(x_raw))

    x_raw = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(x_raw)

    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw)
    print("Leaving function load_data")
    return x, y, vocabulary, vocabulary_inv, df, labels  # DATA END


def train_cnn_rnn():  # TRAIN
    print("Entering function train_cnn_rnn")
    x_, y_, vocabulary, vocabulary_inv, df, labels = load_data(TRAIN_FILE_PATH)
    # Assign a 300 dimension vector to each word
    word_embeddings = load_embeddings(vocabulary)
    embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
    embedding_mat = np.array(embedding_mat, dtype=np.float32)

    # Split the original dataset into train set and test set
    x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1)

    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)

    print('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    print('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    # Create a directory, everything related to the training will be saved in this directory
    timestamp = str(int(time.time()))
    if USE_TMP_FOLDER:
        timestamp = "temp"
    trained_dir = PRO_FLD + 'trained_results_' + timestamp + '/'
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = TextCNNRNN(
                embedding_mat=embedding_mat,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                non_static=params['non_static'],
                hidden_unit=params['hidden_unit'],
                max_pool_size=params['max_pool_size'],
                filter_sizes=map(int, params['filter_sizes'].split(",")),
                num_filters=params['num_filters'],
                embedding_size=params['embedding_dim'],
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name='global_step', trainable=False)
            # optimizer = tf.train.RMSPropOptimizer(0.001, decay=0.9)
            optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
            grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Checkpoint files will be saved in this directory during training
            checkpoint_dir = PRO_FLD + 'checkpoints_' + timestamp + '/'
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            def real_len(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                _, _, _, _ = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)
                return

            def dev_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: 1.0,
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                loss_l, accuracy_l, num_correct, predictions_l = sess.run(
                    [cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
                return accuracy_l, loss_l, num_correct, predictions_l

            def print_stats(stat_dict_total, stat_dict_correct):
                longest_key = 0
                for key in stat_dict_total:
                    if len(key) > longest_key:
                        longest_key = len(key)
                for key in stat_dict_total:
                    my_msg = "     Class {:{}s}: ({}/{}) -> accuracy: {:.4f}%"
                    temp = 0
                    if key in stat_dict_correct:
                        temp = stat_dict_correct[key]
                    my_acc_l = (float(temp) / float(stat_dict_total[key]))*100
                    print(my_msg.format(key, longest_key, temp, stat_dict_total[key], my_acc_l))
                return

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
            best_accuracy, best_at_step, current_step = 0, 0, 0
            number_of_steps_in_total = int((len(x_train) / params['batch_size'] + 1)*params['num_epochs'])  # steps
            print("***There will be {} steps total".format(number_of_steps_in_total))
            stat_dict_all_total, stat_dict_all_correct = defaultdict(int), defaultdict(int)
            # Train the model with x_train and y_train
            for train_batch in train_batches:
                stat_dict_step_total, stat_dict_step_correct = defaultdict(int), defaultdict(int)
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                # Evaluate the model with x_dev and y_dev
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)

                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
                        ind = 0
                        for p in predictions:
                            real_class_value = int(np.argmax(y_dev_batch[ind]))
                            real_class_label = labels[real_class_value]
                            stat_dict_step_total[real_class_label] += 1
                            if p == real_class_value:
                                stat_dict_step_correct[real_class_label] += 1
                            ind += 1
                        total_dev_correct += num_dev_correct
                    accuracy = float(total_dev_correct) / len(y_dev)

                    # Stats prints
                    mes = "STEP {} - ({}/{}) -> accuracy: {:.4f}%"
                    print(mes.format(current_step, int(total_dev_correct), len(y_dev), accuracy*100))
                    if current_step % PRINT_CLASSES_STATS_EACH_X_STEPS == 0:
                        print_stats(stat_dict_step_total, stat_dict_step_correct)

                    if accuracy > best_accuracy:
                        best_accuracy, best_at_step = accuracy, current_step
                        if SHOULD_SAVE:
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            logging.info('    Saved model {} at step {}'.format(path, best_at_step))
                        msg = '    Best accuracy {:.4f}% at step {}/{} ({}/{})'
                        logging.info(msg.format(best_accuracy*100, best_at_step, number_of_steps_in_total,
                                                int(total_dev_correct), len(y_dev)))
                stat_dict_all_total = dict(Counter(stat_dict_all_total)+Counter(stat_dict_step_total))
                stat_dict_all_correct = dict(Counter(stat_dict_all_correct)+Counter(stat_dict_step_correct))
            train_msg = '***Training is complete. Best accuracy {:.4f}% at step {}/{}'
            print(train_msg.format(best_accuracy*100, best_at_step, current_step))
            # Stats prints
            print_stats(stat_dict_all_total, stat_dict_all_correct)
            # Save the model files to trained_dir. predict.py needs trained model files.
            if SHOULD_SAVE:
                saver.save(sess, trained_dir + "best_model.ckpt")

            # Evaluate x_test and y_test
            if RUN_TEST_AFTER_TRAIN:
                print('***Testing...')
                saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
                test_batches = batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
                total_test_correct = 0
                test_stat_dict_total, test_dict_correct = defaultdict(int), defaultdict(int)
                for test_batch in test_batches:
                    x_test_batch, y_test_batch = zip(*test_batch)
                    acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
                    ind = 0
                    for p in predictions:
                        real_class_value = int(np.argmax(y_test_batch[ind]))
                        real_class_label = labels[real_class_value]
                        test_stat_dict_total[real_class_label] += 1
                        if p == real_class_value:
                            test_dict_correct[real_class_label] += 1
                        ind += 1
                    total_test_correct += int(num_test_correct)
                my_acc = (float(total_test_correct) / float(len(y_test))) * 100
                acc_msg = 'Accuracy on test set - ({}/{}) -> accuracy: {:.4f}%'
                print(acc_msg.format(total_test_correct, len(y_test), my_acc))
                # Stats prints
                print_stats(test_stat_dict_total, test_dict_correct)
                if PRINT_WORD_PARAGRAPH:
                    mdiff = 'data file={}. us and spain 45-150 tokens. BasicLSTMCell'.format(CSV_FULL_PATH)
                    last_out = 7
                    print('Difference from out{}: {}'.format(last_out, mdiff))
                    m1 = 'Training best acc {:.4f}% at step {}/{}'
                    print(m1.format(best_accuracy*100, best_at_step, current_step))
                    m2 = 'Test results: Accuracy on test set - ({}/{}) -> accuracy: {:.4f}%'
                    print(m2.format(total_test_correct, len(y_test), my_acc))
                    print_stats(test_stat_dict_total, test_dict_correct)

    # # Save trained parameters and files since predict.py needs them
    # with open(trained_dir + 'words_index.json', 'w') as outfile:
    #     json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
    # with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
    #     pickle.dump(embedding_mat, outfile)
    # with open(trained_dir + 'labels.json', 'w') as outfile:
    #     json.dump(labels, outfile, indent=4, ensure_ascii=False)
    #
    # params['sequence_length'] = x_train.shape[1]
    # with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
    #     json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)
    print("Leaving function train_cnn_rnn")
    return


def make_txt_csv_zip():
    make_txt_file()
    all_rows, max_rows, file = [], -1, REGULAR_FILE_TO_CSV
    f = open(file, 'r', encoding="utf8")
    for line in f:
        if 0 < max_rows <= len(all_rows):  # -1: read all file
            break
        line = line.strip()
        line = line.replace("\n", "")
        line = line.replace("\t", "")
        all_rows.append(line.lower())
    f.close()
    all_data_x, all_data_y = [], []
    for row in all_rows:
        sub_string = row[row.find('[')+1:row.find(']')]
        all_data_y.append(sub_string)
        sub_string = row[row.find('['):row.find(']') + 1]
        row = row.replace(sub_string, '', 1)
        row = row.strip()
        all_data_x.append(row)

    with open(CSV_FULL_PATH, 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Descript"])
        writer.writerows(zip(all_data_y, all_data_x))

    zf = zipfile.ZipFile(CSV_FULL_PATH + ".zip", "w")
    zf.write(CSV_FULL_PATH, CSV_NAME)
    zf.close()
    return


def read_raw_file_to_list(file, max_rows, label):
    my_list = []
    f = open(file, 'r', encoding="utf8")
    # max_words_in_sen = 0
    # sen = ''
    for line in f:
        if 0 < max_rows <= len(my_list):  # -1: read all file
            break
        sub_string = line[line.find('['):line.find(']') + 1]
        line = line.replace(sub_string, '', 1)
        sub_string = line[line.find('['):line.find(']') + 1]
        line = line.replace(sub_string, '', 1)
        line = clean_str(line)
        line = line.strip()

        bad_sentence = False
        word_list = line.split()
        for w in word_list:
            if word_list.count(w) > COUNT_WORD:
                bad_sentence = True

        line_word_count = len(line.split())
        if MAXIMUM_ROW_LENGTH >= line_word_count >= MINIMUM_ROW_LENGTH and not bad_sentence:
            line = ("[" + label + "] " + line)
            my_list.append(line.lower())
            # if line_word_count > max_words_in_sen:
            #     max_words_in_sen = line_word_count
            #     sen = line

    # print(max_words_in_sen)
    # print(sen)
    f.close()
    return my_list


def make_txt_file():
    print("Need to parse raw data")
    print("    Parsing raw data...")
    base_path_nn = RAW_DATA_PATH + REDDIT_DIR + NON_NATIVE_RAW_FOLDER_NAME
    non_native_file_names_all = os.listdir(base_path_nn)
    non_native_file_names = []
    nn_list = ['spain.txt']
    for s in non_native_file_names_all:
        for nn_country_name in nn_list:
            if nn_country_name in s.lower():
                non_native_file_names.append(s)

    base_path_na = RAW_DATA_PATH + REDDIT_DIR + NATIVE_RAW_FOLDER_NAME
    native_file_names_all = os.listdir(base_path_na)
    native_file_names = []
    na_list = ['us.txt']
    for s in native_file_names_all:
        for na_country_name in na_list:
            if na_country_name in s.lower():
                native_file_names.append(s)

    print("    target files: {},{}".format(non_native_file_names, native_file_names))
    # sys.exit(0) # recommended to see debug before moving forward
    class_size = 11044
    all_semi_raw_data = []
    for file in non_native_file_names:
        local_list = read_raw_file_to_list(base_path_nn + file, class_size, file.split('.')[1])
        all_semi_raw_data += local_list

    for file in native_file_names:
        local_list = read_raw_file_to_list(base_path_na + file, class_size, file.split('.')[1])
        all_semi_raw_data += local_list

    np.random.shuffle(all_semi_raw_data)
    print("    all_data size is {}".format(len(all_semi_raw_data)))
    print("    saving to file: {}".format(REGULAR_FILE_TO_CSV))

    dst_file = open(REGULAR_FILE_TO_CSV, 'w+', encoding="utf8")
    for line in all_semi_raw_data:
        dst_file.write(line + '\n')
    dst_file.close()
    print("    file {} with {} lines was created".format(REGULAR_FILE_TO_CSV, len(all_semi_raw_data)))
    print("    Finished Parsing raw data")
    return


def args_print(stage, duration=0):
    print("{} ----------------------".format(stage))
    print("epochs {}".format(params['num_epochs']))
    print("batchSize {}".format(params['batch_size']))
    print("dropout_keep_prob {}".format(params['dropout_keep_prob']))
    print("embedding_dim {}".format(params['embedding_dim']))
    print("evaluate_every {}".format(params['evaluate_every']))
    print("num_filters {}".format(params['num_filters']))
    print("filter_sizes {}".format(params['filter_sizes']))
    print("hidden_unit {}".format(params['hidden_unit']))
    print("l2_reg_lambda {}".format(params['l2_reg_lambda']))
    print("max_pool_size {}".format(params['max_pool_size']))
    print("non_static {}".format(params['non_static']))
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time(HH:MM:SS): {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))
    return


if __name__ == '__main__':
    print("Entering function __main__")
    if MAKE_NEW_DATA_FILE:
        make_txt_csv_zip()
    else:
        total_start_time = time.time()
        if IS_TRAIN:
            train_cnn_rnn()
        else:
            # predict_unseen_data()
            pass
        dur = time.time() - total_start_time
        args_print('params', int(dur))
    print("Leaving function __main__")


# def load_trained_params(trained_dir):
#     print("Entering function load_trained_params")
#     params = json.loads(open(trained_dir + 'trained_parameters.json').read())
#     words_index = json.loads(open(trained_dir + 'words_index.json').read())
#     labels = json.loads(open(trained_dir + 'labels.json').read())
#
#     with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
#         fetched_embedding = pickle.load(input_file)
#     embedding_mat = np.array(fetched_embedding, dtype=np.float32)
#     print("Leaving function load_trained_params")
#     return params, words_index, labels, embedding_mat


# def load_test_data(test_file, labels):
#     print("Entering function load_test_data")
#     df = pd.read_csv(test_file)
#     select = ['Descript']
#
#     df = df.dropna(axis=0, how='any', subset=['Descript'])
#     test_examples = df[select[0]].apply(lambda x: clean_str(x).split(' ')).tolist()
#
#     num_labels = len(labels)
#     one_hot = np.zeros((num_labels, num_labels), int)
#     np.fill_diagonal(one_hot, 1)
#     label_dict = dict(zip(labels, one_hot))
#
#     y_ = None
#     if 'Category' in df.columns:
#         select.append('Category')
#         y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()
#
#     not_select = list(set(df.columns) - set(select))
#     df = df.drop(not_select, axis=1)
#     print("Leaving function load_test_data")
#     return test_examples, y_, df


# def map_word_to_index(examples, words_index):
#     x_ = []
#     for example in examples:
#         temp = []
#         for word in example:
#             if word in words_index:
#                 temp.append(words_index[word])
#             else:
#                 temp.append(0)
#         x_.append(temp)
#     return x_


# def predict_unseen_data():
#     print("Entering function predict_unseen_data")
#     trained_dir = PRO_FLD + TRA_FLD
#     print(trained_dir)
#     if not trained_dir.endswith('/'):
#         trained_dir += '/'
#     test_file = PRO_FLD + 'data/train.csv.zip'
#
#     params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
#     x_, y_, df = load_test_data(test_file, labels)
#     x_ = pad_sentences(x_, forced_sequence_length=params['sequence_length'])
#     x_ = map_word_to_index(x_, words_index)
#
#     x_test, y_test = np.asarray(x_), None
#     if y_ is not None:
#         y_test = np.asarray(y_)
#
#     timestamp = trained_dir.split('/')[-2].split('_')[-1] # check when using TEMP FOLDER
#     predicted_dir = PRO_FLD + 'predicted_results_' + timestamp + '/'
#     if os.path.exists(predicted_dir):
#         shutil.rmtree(predicted_dir)
#     os.makedirs(predicted_dir)
#
#     with tf.Graph().as_default():
#         session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             cnn_rnn = TextCNNRNN(
#                 embedding_mat=embedding_mat,
#                 non_static=params['non_static'],
#                 hidden_unit=params['hidden_unit'],
#                 sequence_length=len(x_test[0]),
#                 max_pool_size=params['max_pool_size'],
#                 filter_sizes=map(int, params['filter_sizes'].split(",")),
#                 num_filters=params['num_filters'],
#                 num_classes=len(labels),
#                 embedding_size=params['embedding_dim'],
#                 l2_reg_lambda=params['l2_reg_lambda'])
#
#             def real_len(batches):
#                 return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]
#
#             def predict_step(x_batch):
#                 feed_dict = {
#                     cnn_rnn.input_x: x_batch,
#                     cnn_rnn.dropout_keep_prob: 1.0,
#                     cnn_rnn.batch_size: len(x_batch),
#                     cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
#                     cnn_rnn.real_len: real_len(x_batch),
#                 }
#                 predictions = sess.run([cnn_rnn.predictions], feed_dict)
#                 return predictions
#
#             checkpoint_file = trained_dir + 'best_model.ckpt'
#             saver = tf.train.Saver(tf.all_variables())
#             saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#             saver.restore(sess, checkpoint_file)
#             print('{} has been loaded'.format(checkpoint_file))
#
#             batches = batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
#
#             predictions, predict_labels = [], []
#             for x_batch in batches:
#                 batch_predictions = predict_step(x_batch)[0]
#                 for batch_prediction in batch_predictions:
#                     predictions.append(batch_prediction)
#                     predict_labels.append(labels[batch_prediction])
#
#             # Save the predictions back to file
#             df['NEW_PREDICTED'] = predict_labels
#             columns = sorted(df.columns, reverse=True)
#             df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')
#
#             if y_test is not None:
#                 y_test = np.array(np.argmax(y_test, axis=1))
#                 accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
#                 print('The prediction accuracy is: {}'.format(accuracy))
#
#             print('Prediction is complete, all files have been saved: {}'.format(predicted_dir))
#     print("Leaving function predict_unseen_data")
#     return
