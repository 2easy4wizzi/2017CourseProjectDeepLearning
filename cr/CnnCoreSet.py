import pickle
import os
import urllib
import tarfile
import zipfile
import sys
import tensorflow as tf
import numpy as np
from time import time
import math
import matplotlib.pyplot as plt
import urllib.request
from collections import defaultdict
import shutil

# %matplotlib inline

PRJ_DIR = '../'
SRC_DIR = PRJ_DIR + 'src/'
DAT_DIR = PRJ_DIR + 'data_set/'
CIF_DIR = DAT_DIR + 'cifar_10/'
OPT_STR = "AdamOptimizer"
SAVE_PATH = PRJ_DIR + "tempModels/"
SAVE_MODEL_NAME = '{}42350Params/model.ckpt'.format(OPT_STR)

RUN_TRAIN = False  # to load model and test only the test set - set on FALSE
RUN_TEST = True  # to load model and test only the test set - set on FALSE
ADAM_OPTIMIZER = True  # what model will be selected
SHOULD_SAVE_MODELS = False

SHOULD_PRINT_SHAPES = True
SHOULD_PRINT_VARS_NAMES = True
SHOULD_PRINT_START_VARS = True
SHOULD_PRINT_END_VARS = True
_BATCH_SIZE = 196
_EPOCH = 5
_TRAIN_KEEP_PROB = 0.5
X_TO_FIRST_CONV_MAPS = 0
FIRST_CONV_TO_SECOND_CONV_MAPS = 30
SECOND_CONV_TO_THIRD_CONV_MAPS = 40
THIRD_CONV_TO_FOURTH_CONV_MAPS = 95
_IMAGE_SIZE = 32
_IMAGE_CHANNELS = 3
_NUM_CLASSES = 10
DASHED = '###########################################################################################################'


def print_variable_val(var):
    print(var.name, var.eval())
    return


def weight_variable(shape_local, _name):
    initial = tf.truncated_normal(shape_local, stddev=0.1)
    return tf.Variable(initial, name=_name)


def bias_variable(shape_local):
    initial = tf.constant(0.1, shape=shape_local)
    return tf.Variable(initial, name="bias_variable")


def conv2d(x_local, w_local):
    return tf.nn.conv2d(x_local, w_local, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x_local):
    return tf.nn.max_pool(x_local, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def model():
    tf.reset_default_graph()
    epsilon = 1e-3

    x_local = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
    y_local = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
    x_image = tf.reshape(x_local, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
    if SHOULD_PRINT_SHAPES and RUN_TRAIN:
        print("x shape: {}".format(x_local.get_shape()))
        print("y shape: {}".format(y_local.get_shape()))
        print("x_image shape: {}".format(x_image.get_shape()))
    keep_prob_feed = tf.placeholder(tf.float32, name="keepProbVar")

    # first layer - conv
    w1 = weight_variable([3, 3, _IMAGE_CHANNELS, FIRST_CONV_TO_SECOND_CONV_MAPS], 'w_1')
    z1_bn = conv2d(x_image, w1)
    batch_mean1, batch_var1 = tf.nn.moments(z1_bn, [0])
    scale1 = tf.Variable(tf.ones([FIRST_CONV_TO_SECOND_CONV_MAPS]), name="bn_scale1")
    beta1 = tf.Variable(tf.zeros([FIRST_CONV_TO_SECOND_CONV_MAPS]), name="bn_beta1")
    bn1 = tf.nn.batch_normalization(z1_bn, batch_mean1, batch_var1, beta1, scale1, epsilon)
    conv1 = tf.nn.relu(bn1)
    pool_1 = max_pool_2x2(conv1)
    drop_1 = tf.nn.dropout(pool_1, keep_prob_feed)
    if SHOULD_PRINT_SHAPES and RUN_TRAIN:
        print("conv1 shape: {}".format(conv1.get_shape()))
        print("pool_1 shape: {}".format(pool_1.get_shape()))
        print("drop_1 shape: {}".format(drop_1.get_shape()))

    # second layer - conv
    w2 = weight_variable([3, 3, FIRST_CONV_TO_SECOND_CONV_MAPS, SECOND_CONV_TO_THIRD_CONV_MAPS], 'w_2')
    z2_bn = conv2d(drop_1, w2)
    batch_mean2, batch_var2 = tf.nn.moments(z2_bn, [0])
    scale2 = tf.Variable(tf.ones([SECOND_CONV_TO_THIRD_CONV_MAPS]), name="bn_scale2")
    beta2 = tf.Variable(tf.zeros([SECOND_CONV_TO_THIRD_CONV_MAPS]), name="bn_beta2")
    bn2 = tf.nn.batch_normalization(z2_bn, batch_mean2, batch_var2, beta2, scale2, epsilon)
    conv2 = tf.nn.relu(bn2)
    pool_2 = max_pool_2x2(conv2)
    if SHOULD_PRINT_SHAPES and RUN_TRAIN:
        print("conv2 shape: {}".format(conv2.get_shape()))
        print("pool_2 shape: {}".format(pool_2.get_shape()))

    # third layer - conv
    w3 = weight_variable([2, 2, SECOND_CONV_TO_THIRD_CONV_MAPS, THIRD_CONV_TO_FOURTH_CONV_MAPS], 'w_3')
    z3_bn = conv2d(pool_2, w3)
    batch_mean3, batch_var3 = tf.nn.moments(z3_bn, [0])
    scale3 = tf.Variable(tf.ones([THIRD_CONV_TO_FOURTH_CONV_MAPS]), name="bn_scale3")
    beta3 = tf.Variable(tf.zeros([THIRD_CONV_TO_FOURTH_CONV_MAPS]), name="bn_beta3")
    bn3 = tf.nn.batch_normalization(z3_bn, batch_mean3, batch_var3, beta3, scale3, epsilon)
    conv3 = tf.nn.relu(bn3)
    pool_3 = max_pool_2x2(conv3)
    drop_3 = tf.nn.dropout(pool_3, keep_prob_feed)
    shapedrop3, flat_size, first = drop_3.get_shape(), 1, True
    for mydim in shapedrop3:
        if first is False:
            flat_size *= mydim.value
        first = False
    flat = tf.reshape(drop_3, [-1, flat_size])
    if SHOULD_PRINT_SHAPES and RUN_TRAIN:
        print("conv3 shape: {}".format(conv3.get_shape()))
        print("pool_3 shape: {}".format(pool_3.get_shape()))
        print("drop_3 shape: {}".format(drop_3.get_shape()))
        print("flat shape: {}".format(flat.get_shape()))

    softmax = tf.nn.softmax(tf.layers.dense(inputs=flat, units=_NUM_CLASSES))
    if SHOULD_PRINT_SHAPES and RUN_TRAIN:
        print("softmax shape: {}".format(softmax.get_shape()))
    y_pred_cls_l = tf.argmax(softmax, axis=1)
    loss_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y_local))
    if ADAM_OPTIMIZER:
        optimizer_l = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss_l)
    else:
        optimizer_l = tf.train.GradientDescentOptimizer(0.5).minimize(loss_l)
    # PREDICTION AND ACCURACY CALCULATION
    correct_prediction_l = tf.equal(y_pred_cls_l, tf.argmax(y_local, axis=1))
    accuracy_l = tf.reduce_mean(tf.cast(correct_prediction_l, tf.float32))

    return x_local, y_local, loss_l, optimizer_l, correct_prediction_l, accuracy_l, y_pred_cls_l, keep_prob_feed


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()
    return


def maybe_download_and_extract():
    main_directory = DAT_DIR
    cifar_10_directory = CIF_DIR
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        # data = urllib.request.urlretrieve("http://...")
        file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory + "./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)
    return


def get_data_set(name="train"):
    x_local = None
    y_local = None

    maybe_download_and_extract()

    f = open(CIF_DIR + '/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open(CIF_DIR + '/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32 * 32 * 3)

            if x_local is None:
                x_local = _X
                y_local = _Y
            else:
                x_local = np.concatenate((x_local, _X), axis=0)
                y_local = np.concatenate((y_local, _Y), axis=0)

    elif name is "test":
        f = open(CIF_DIR + '/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x_local = datadict["data"]
        y_local = np.array(datadict['labels'])

        x_local = np.array(x_local, dtype=float) / 255.0
        x_local = x_local.reshape([-1, 3, 32, 32])
        x_local = x_local.transpose([0, 2, 3, 1])
        x_local = x_local.reshape(-1, 32 * 32 * 3)

    return x_local, dense_to_one_hot(y_local)


def pre():
    global sess
    sess = tf.Session()

    global train_x, train_y
    if RUN_TRAIN:
        train_x, train_y = get_data_set("train")
        # a = 3
        # pic_x = train_x[a]
        # pic_x = pic_x.reshape(32, 32, 3)
        # print("1 pic dims: {}".format(pic_x.shape))
        # print(train_y[a])
        # plt.imshow(pic_x)
        # plt.show()
        # sys.exit(0)

    global test_x, test_y
    test_x, test_y = get_data_set("test")

    global x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob2
    x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob2 = model()
    global total_parameters, global_accuracy, best_epoch, train_error_list, test_error_list
    global_accuracy = 0
    best_epoch = 0
    train_error_list = []
    test_error_list = []

    if RUN_TRAIN:
        global BATCHSIZE, _STEPS_PRINT
        BATCHSIZE = int(math.ceil(len(train_x) / _BATCH_SIZE))
        _STEPS_PRINT = (BATCHSIZE / 2 - 1)

    global init, saver
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    global total_parameters
    total_parameters = 0
    if RUN_TRAIN:
        if SHOULD_PRINT_VARS_NAMES:
            print("---vars name and shapes---")
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            if SHOULD_PRINT_VARS_NAMES:
                print(variable.name, shape, variable_parameters)
            total_parameters += variable_parameters
        if SHOULD_PRINT_VARS_NAMES:
            print("total PARAM {:,}".format(total_parameters))
            print("---done vars---")
    return


def args_print(stage, duration=0):
    print("{} ----------------------".format(stage))
    print("epochs {}".format(_EPOCH))
    print("batchSize {}".format(_BATCH_SIZE))
    print("keepProb {}".format(_TRAIN_KEEP_PROB))
    print("train_x dims: {}".format(train_x.shape))
    print("total PARAM {:,}".format(total_parameters))
    # print("X_TO_FIRST_CONV_MAPS {}".format(X_TO_FIRST_CONV_MAPS))
    print("FIRST_CONV_TO_SECOND_CONV_MAPS {}".format(FIRST_CONV_TO_SECOND_CONV_MAPS))
    print("SECOND_CONV_TO_THIRD_CONV_MAPS {}".format(SECOND_CONV_TO_THIRD_CONV_MAPS))
    print("THIRD_CONV_TO_FOURTH_CONV_MAPS {}".format(THIRD_CONV_TO_FOURTH_CONV_MAPS))
    print("global_accuracy {}".format(global_accuracy))
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("duration(formatted HH:MM:SS): {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))


def main():
    pre()
    if RUN_TRAIN:
        if SHOULD_PRINT_START_VARS:
            args_print("start")
        total_start_time = time()
        if SHOULD_SAVE_MODELS:
            if os.path.exists(SAVE_PATH):
                shutil.rmtree(SAVE_PATH)
        for i in range(_EPOCH):
            print(DASHED)
            print("Epoch: {0}/{1}".format((i + 1), _EPOCH))
            train(i)
        print(DASHED)
        duration = time() - total_start_time
        if SHOULD_PRINT_END_VARS:
            args_print("end", int(duration))

        msg = "optimizer is {}, best accuracy = {}% achieved on epoch number {}"
        print(msg.format(OPT_STR, global_accuracy, best_epoch))
        # print_graph('error/epochs(train in red, test in green)', 'epochs', 'error', train_error_list, test_error_list)
    if RUN_TEST:
        load_model_and_run_test()


def print_graph(title, xlabel, ylabel, func_a_data, func_b_data):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(func_a_data, 'r')
    plt.plot(func_b_data, 'g')
    plt.show()


def load_model_and_run_test():

    with tf.Session() as sess2:
        sess2.run(tf.global_variables_initializer())

        # #this network variables
        # w_1 = tf.get_variable("w_1", shape=[3, 3, 3, 30])
        # bn_scale1 = tf.get_variable("bn_scale1", shape=[30])
        # bn_beta1 = tf.get_variable("bn_beta1", shape=[30])
        # w_2 = tf.get_variable("w_2", shape=[3, 3, 30, 40])
        # bn_scale2 = tf.get_variable("bn_scale2", shape=[40])
        # bn_beta2 = tf.get_variable("bn_beta2", shape=[40])
        # w_3 = tf.get_variable("w_3", shape=[2, 2, 40, 95])
        # bn_scale3 = tf.get_variable("bn_scale3", shape=[95])
        # bn_beta3 = tf.get_variable("bn_beta3", shape=[95])
        # dense_kernel = tf.get_variable("dense/kernel", shape=[1520, 10])
        # dense_bias = tf.get_variable("dense/bias", shape=[10])

        saver2 = tf.train.Saver()
        saver2.restore(sess2, SAVE_PATH + SAVE_MODEL_NAME)
        print("Model restored from file: %s" % SAVE_PATH + SAVE_MODEL_NAME)
        print("Checking test_x and test_y on best model with {}".format(OPT_STR))

        # # the way to print variables
        # print_variable_val(w_1)
        # print_variable_val(bn_scale1)
        # print_variable_val(bn_beta1)
        # print_variable_val(w_2)
        # print_variable_val(bn_scale2)
        # print_variable_val(bn_beta2)
        # print_variable_val(w_3)
        # print_variable_val(bn_scale3)
        # print_variable_val(bn_beta3)
        # print_variable_val(dense_kernel)
        # print_variable_val(dense_bias)

        i = 0
        predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
        while i < len(test_x):
            j = min(i + _BATCH_SIZE, len(test_x))
            batch_xs = test_x[i:j, :]
            batch_ys = test_y[i:j, :]
            predicted_class[i:j] = sess2.run(
                y_pred_cls,
                feed_dict={x: batch_xs, y: batch_ys, keep_prob2: 1}
            )
            i = j

        correct = (np.argmax(test_y, axis=1) == predicted_class)
        acc = correct.mean() * 100
        correct_numbers = correct.sum()

        mes = "\nmodel - accuracy: {:.2f}% ({}/{})"
        print(mes.format(acc, correct_numbers, len(test_x)))


def train(epoch):
    global train_error_list
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    current_error = 0
    for s in range(batch_size):
        batch_xs = train_x[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
        batch_ys = train_y[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]

        _, batch_loss, batch_acc = sess.run(
            [optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, keep_prob2: _TRAIN_KEEP_PROB})

        # original: if s % 10
        if s % _STEPS_PRINT == 0:
            mes = "     train STEP {}/{}: accuracy = {:.4f}% , loss = {:.4f}"
            print(mes.format(s, batch_size, batch_acc * 100, batch_loss))

        current_error += 1-batch_acc

    # print("loss avg for epoch {} is {}".format(epoch, current_loss / batch_size))
    train_error_list.append(current_error / batch_size)
    test_and_save(epoch)


def test_and_save(epoch):
    print("Testing on dev...")
    global global_accuracy
    global test_error_list
    global best_epoch
    global saver

    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    current_test_loss, i = 0, 0

    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j], batch_test_loss = sess.run(
            [y_pred_cls, loss], feed_dict={x: batch_xs, y: batch_ys, keep_prob2: 1})
        i = j
        current_test_loss += batch_test_loss

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    test_error_list.append(1-correct.mean())
    acc = correct.mean() * 100
    correct_numbers = correct.sum()

    mes = "     Epoch {} - accuracy: {:.2f}% ({}/{}). BEST STATS: global best acc on epoch {} = {:.2f}"
    print(mes.format((epoch + 1), acc, correct_numbers, len(test_x), best_epoch, global_accuracy))

    if acc > global_accuracy:
        mes = "     Epoch {} receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(epoch+1, acc, global_accuracy))
        if SHOULD_SAVE_MODELS:
            if os.path.exists(SAVE_PATH):
                shutil.rmtree(SAVE_PATH)
            os.makedirs(SAVE_PATH)
            save_path = saver.save(sess, SAVE_PATH + SAVE_MODEL_NAME)
            print("     Model saved in file: %s" % save_path)
        global_accuracy = acc
        best_epoch = epoch + 1
    print_stats(predicted_class)


def print_stats(predicted_class_l):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    stat_dict_step_total, stat_dict_step_correct = defaultdict(int), defaultdict(int)
    ind = 0
    for p in predicted_class_l:
        real_class_value = int(np.argmax(test_y[ind]))
        real_class_label = labels[real_class_value]
        stat_dict_step_total[real_class_label] += 1
        if p == real_class_value:
            stat_dict_step_correct[real_class_label] += 1
        ind += 1
    longest_key = 0
    for key in stat_dict_step_total:
        if len(key) > longest_key:
            longest_key = len(key)
    for key in stat_dict_step_total:
        my_msg = "          Class {:{}s}: ({}/{}) -> accuracy: {:.4f}%"
        temp = 0
        if key in stat_dict_step_correct:
            temp = stat_dict_step_correct[key]
        my_acc_l = (float(temp) / float(stat_dict_step_total[key])) * 100
        print(my_msg.format(key, longest_key, temp, stat_dict_step_total[key], my_acc_l))
    return


if __name__ == "__main__":
    main()
