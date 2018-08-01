import re
import itertools
import os
import time
import shutil
import pickle
import pandas as pd
import json
from collections import Counter
import numpy as np
import tensorflow as tf
# pip.main(['install', 'builtins '])
# from builtins import enumerate
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, './cr/')
from text_cnn_rnn import TextCNNRNN

import logging
logging.getLogger().setLevel(logging.INFO)

PRO_FLD = ''
TRA_FLD = 'trained_results_1533109035/'
IS_TRAIN = True
TRAIN_FILE_PATH = PRO_FLD + 'data/parsed_input.csv.zip'
TRAIN_FILE_PATH = PRO_FLD + 'data/shortdata.csv.zip'

params = {}
params['batch_size'] = 128
params['dropout_keep_prob'] = 0.5
params['embedding_dim'] = 300
params['evaluate_every'] = 1
params['filter_sizes'] = "3,4,5"
params['hidden_unit'] = 300
params['l2_reg_lambda'] = 0.0
params['max_pool_size'] = 4
params['non_static'] = False
params['num_epochs'] = 1
params['num_filters'] = 32


def clean_str(s):  # DATA
    s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
    s = re.sub(r" : ", ":", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()


def load_embeddings(vocabulary):
    # print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    # print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    # print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
    """Pad setences during training or prediction"""
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
    print('The sentences length (after padding) will be {}(length of the longest sentence)'.format(sequence_length))
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
    # print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
    return padded_sentences


def build_vocab(sentences):
    # print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    # print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
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
    print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
    df = pd.read_csv(filename, compression='zip')
    selected = ['Category', 'Descript']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))

    labels = sorted(list(set(df[selected[0]].tolist())))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw = df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    x_raw = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(x_raw)

    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw)
    print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
    return x, y, vocabulary, vocabulary_inv, df, labels  # DATA END


def train_cnn_rnn():  # TRAIN
    print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
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
            optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
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
                _, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy],
                                                   feed_dict)

            def dev_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: 1.0,
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                step, loss, accuracy, num_correct, predictions = sess.run(
                    [global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions],
                    feed_dict)
                return accuracy, loss, num_correct, predictions

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
            best_accuracy, best_at_step = 0, 0
            number_of_steps_in_total = len(x_train) / 128 + 1  # steps
            number_of_steps_in_total *= params['num_epochs']
            logging.info("---There will be {} steps total".format(number_of_steps_in_total))
            # Train the model with x_train and y_train
            for train_batch in train_batches:
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
                        count = 0
                        count_good = 0
                        for p in predictions:
                            # if p == y_dev_batch[count]:
                            #     count_good += 1
                            print(p)
                            print(y_dev_batch[count])
                            break
                            count += 1
                        print(acc, num_dev_correct, count_good, count)
                        total_dev_correct += num_dev_correct
                    accuracy = float(total_dev_correct) / len(y_dev)
                    print('Step {} - Accuracy on dev set: {}'.format(current_step, accuracy))

                    if accuracy > best_accuracy:
                        best_accuracy, best_at_step = accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.info('    Saved model {} at step {}'.format(path, best_at_step))
                        logging.info('    Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
            print('Training is complete, testing the best model on x_test and y_test')

            # Save the model files to trained_dir. predict.py needs trained model files.
            saver.save(sess, trained_dir + "best_model.ckpt")

            # Evaluate x_test and y_test
            saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            test_batches = batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
                total_test_correct += int(num_test_correct)
            print('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))

    # Save trained parameters and files since predict.py needs them
    with open(trained_dir + 'words_index.json', 'w') as outfile:
        json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
    with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
        pickle.dump(embedding_mat, outfile)
    with open(trained_dir + 'labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4, ensure_ascii=False)

    params['sequence_length'] = x_train.shape[1]
    with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)
    print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
    return


# def load_trained_params(trained_dir):
#     print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
#     params = json.loads(open(trained_dir + 'trained_parameters.json').read())
#     words_index = json.loads(open(trained_dir + 'words_index.json').read())
#     labels = json.loads(open(trained_dir + 'labels.json').read())
#
#     with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
#         fetched_embedding = pickle.load(input_file)
#     embedding_mat = np.array(fetched_embedding, dtype=np.float32)
#     print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
#     return params, words_index, labels, embedding_mat


# def load_test_data(test_file, labels):
#     print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
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
#     print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
#     return test_examples, y_, df


# def map_word_to_index(examples, words_index):
#     print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
#     x_ = []
#     for example in examples:
#         temp = []
#         for word in example:
#             if word in words_index:
#                 temp.append(words_index[word])
#             else:
#                 temp.append(0)
#         x_.append(temp)
#     print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
#     return x_


# def predict_unseen_data():
#     print("function {} {}".format(sys._getframe().f_code.co_name, "start"))
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
#     timestamp = trained_dir.split('/')[-2].split('_')[-1]
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
#     print("function {} {}".format(sys._getframe().f_code.co_name, "exit"))
#     return


if __name__ == '__main__':
    print("function {} {}".format(__name__, "start"))
    if IS_TRAIN:
        train_cnn_rnn()
    else:
        pass
        # predict_unseen_data()

    print("function {} {}".format(__name__, "exit"))
