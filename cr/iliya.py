import numpy as np
import tensorflow as tf
import re
from random import randint
import math


def load_embeddings_as_matrix(embeddings_filename):
    words = {}
    embeddings_matrix = []
    with open(embeddings_filename, 'r') as fin:
        for i, line in enumerate(fin.readlines()):
            words[line.split()[0]] = i
            embedding = np.array([float(val) for val in line.split()[1:]], dtype='float32')
            embeddings_matrix.append(embedding)
    # embedding for unk words
    embeddings_matrix.append(np.zeros(embedding_dimension, dtype='float32'))
    return words, np.matrix(embeddings_matrix, dtype='float32')


def clean_sentence(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string)


def load_reviews_data(filename):
    data = []
    labels = []
    counter = 0
    with open(filename, 'r') as fin:
        for line in fin:
            if counter > 100000:
                break
            review_body = line.split('\t')[13]
            review_score = line.split('\t')[7]
            data.append(clean_sentence(review_body))
            labels.append(int(review_score))
            counter += 1
        # end for
    # end with
    return data, labels


def convert_sentence_to_word_indices(text):
    text_indices = np.zeros((max_review_length), dtype='int32')
    for i, token in enumerate(text.split()):
        if i == max_review_length: break
        text_indices[i] = words.get(token, unk_word_index)
    # end for
    return text_indices


def convert_data_to_word_indices(data):
    data_indices = []
    for review in data:
        data_indices.append(convert_sentence_to_word_indices(review))
    # end for
    return np.matrix(data_indices)


def convert_to_array(label):
    array_label = [0] * num_classes
    array_label[label - 1] = 1
    return array_label


def get_batch_sequential(total_examples, batch_num, batch_size):
    batch = tst_data_idx[batch_num * batch_size:(batch_num + 1) * batch_size]
    labels = [convert_to_array(label) for label in tst_labels[batch_num * batch_size:(batch_num + 1) * batch_size]]

    return batch, labels


def get_basic_model(embeddings):
    tf.reset_default_graph()
    input_data = tf.placeholder(tf.int32, [batch_size, max_review_length])
    input_labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    data = tf.Variable(tf.zeros([batch_size, max_review_length, embedding_dimension]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(embeddings, input_data)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob, dtype=tf.float32)
    value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    value = tf.transpose(value, [1, 0, 2])

    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=input_labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, input_labels, keep_prob, optimizer, loss, accuracy


def get_bidirectional_model(embeddings):
    tf.reset_default_graph()
    input_data = tf.placeholder(tf.int32, [batch_size, max_review_length])
    input_labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    data = tf.Variable(tf.zeros([batch_size, max_review_length, embedding_dimension]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(embeddings, input_data)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=keep_prob, dtype=tf.float32)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=keep_prob, dtype=tf.float32)
    value, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, data, dtype=tf.float32)

    value = tf.concat(value, 2)
    value = tf.transpose(value, [1, 0, 2])

    weight = tf.Variable(tf.truncated_normal([2 * lstm_units, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = tf.reduce_mean(tf.squared_difference(prediction, input_labels))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=input_labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, input_labels, keep_prob, optimizer, loss, accuracy


def train_epoch(i, saver):
    global global_accuracy
    total_loss = 0
    total_accuracy = 0
    for b in range(batches):
        # next batch of reviews
        # next_batch, next_batch_labels = get_train_batch(total_examples)
        next_batch, next_batch_labels = get_batch_sequential(total_examples, b, batch_size)
        _, batch_loss, batch_accuracy = sess.run([optimizer, loss, accuracy],
                                                 {input_data: next_batch, input_labels: next_batch_labels,
                                                  keep_prob: 0.75})
        total_accuracy += batch_accuracy
        total_loss += batch_loss
    # end for
    avg_accuracy = float(total_accuracy) / batches
    avg_loss = float(total_loss) / batches

    print('epoch', i, 'average loss and accuracy:', "{0:.3f}".format(avg_loss),
          "{0:.3f}".format(avg_accuracy))

    if avg_accuracy > global_accuracy:
        print('saving current model...')
        save_path = saver.save(sess, "amazon_reviews_data/model-bi.ckpt")
        global_accuracy = avg_accuracy


embedding_dimension = 50
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
glove_embeddings_filename = "amazon_reviews_data/glove.6B.50d.txt"
words, embeddings = load_embeddings_as_matrix(glove_embeddings_filename)

print(len(words), embeddings.shape)
max_review_length = 100
unk_word_index = 400001
filename_tst = "amazon_reviews_data/amazon_reviews_us_Watches_last_10k_tst.tsv"
filename_trn = "amazon_reviews_data/amazon_reviews_us_Watches_v1_00.tsv"
tst_data, tst_labels = load_reviews_data(filename_tst)
trn_data, trn_labels = load_reviews_data(filename_trn)
assert (len(trn_data) == len(trn_labels))
# print(len(trn_data), len(trn_labels))

trn_data_idx = convert_data_to_word_indices(trn_data)
tst_data_idx = convert_data_to_word_indices(tst_data)
print(trn_data_idx.shape, tst_data_idx.shape)

total_examples = trn_data_idx.shape[0]

batch_size = 100
num_classes = 5

total_train_examples = trn_data_idx.shape[0]
# batch, labels = get_train_batch(total_train_examples)
batch, labels = get_batch_sequential(total_train_examples, 3, 100)
print(batch.shape, len(labels))
# print(labels)
lstm_units = 64
input_data, input_labels, keep_prob, optimizer, loss, accuracy = get_bidirectional_model(embeddings)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

batches = int(math.ceil(total_examples / batch_size))
print('number of batches:', batches)

epochs = 100
global_accuracy = 0
for i in range(epochs):
    # print('starting epoch', i, saver)
    train_epoch(i, saver)


sess = tf.InteractiveSession()

saver = tf.train.Saver()
saver.restore(sess, 'amazon_reviews_data/model-bi.ckpt')
print('restored saved session...')

batches = int(math.ceil(len(tst_labels) / batch_size))

total_accuracy = 0
for b in range(batches):
    batch_data, batch_labels = get_batch_sequential(len(tst_labels), b, batch_size)
    batch_accuracy = sess.run(accuracy, {input_data: batch_data, input_labels: batch_labels, keep_prob: 1.0})
    # print('accuracy for batch', b, '{0:.3f}'.format(batch_accuracy))
    total_accuracy += batch_accuracy
# end for

print('test accuracy:', '{0:.3f}'.format(total_accuracy / batches))

