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

def weight_variable(shape_local):
    initial = tf.truncated_normal(shape_local, stddev=0.1)
    return tf.Variable(initial, name="weight_variable")


def bias_variable(shape_local):
    initial = tf.constant(0.1, shape=shape_local)
    return tf.Variable(initial, name="bias_variable")


def conv2d(x_local, w_local):
    return tf.nn.conv2d(x_local, w_local, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x_local):
    return tf.nn.max_pool(x_local, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def gilad_max_pool_kxk(x_local, k=2):
    return tf.nn.max_pool(x_local, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def model():
    epsilon = 1e-3

    x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
    y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
    x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
    if SHOULD_PRINT_SHAPES and TRAIN_MODE:
        print("x shape: {}".format(x.get_shape()))
        print("y shape: {}".format(y.get_shape()))
        print("x_image shape: {}".format(x_image.get_shape()))
    keep_prob2 = tf.placeholder(tf.float32, name="keepProbVar")

    # first layer - conv
    w2 = weight_variable([3, 3, _IMAGE_CHANNELS, FIRST_CONV_TO_SECOND_CONV_MAPS])
    z2_bn = conv2d(x_image, w2)
    batch_mean2, batch_var2 = tf.nn.moments(z2_bn, [0])
    scale2 = tf.Variable(tf.ones([FIRST_CONV_TO_SECOND_CONV_MAPS]), name="scale2")
    beta2 = tf.Variable(tf.zeros([FIRST_CONV_TO_SECOND_CONV_MAPS]), name="beta2")
    bn2 = tf.nn.batch_normalization(z2_bn, batch_mean2, batch_var2, beta2, scale2, epsilon)
    conv2 = tf.nn.relu(bn2)
    pool_1 = max_pool_2x2(conv2)
    drop_1 = tf.nn.dropout(pool_1, keep_prob2)
    if SHOULD_PRINT_SHAPES and TRAIN_MODE:
        print("conv2 shape: {}".format(conv2.get_shape()))
        print("pool_1 shape: {}".format(pool_1.get_shape()))
        print("drop_1 shape: {}".format(drop_1.get_shape()))

    # second layer - conv
    w3 = weight_variable([3, 3, FIRST_CONV_TO_SECOND_CONV_MAPS, SECOND_CONV_TO_THIRD_CONV_MAPS])
    z3_bn = conv2d(drop_1, w3)
    batch_mean3, batch_var3 = tf.nn.moments(z3_bn, [0])
    scale3 = tf.Variable(tf.ones([SECOND_CONV_TO_THIRD_CONV_MAPS]), name="scale3")
    beta3 = tf.Variable(tf.zeros([SECOND_CONV_TO_THIRD_CONV_MAPS]), name="beta3")
    bn3 = tf.nn.batch_normalization(z3_bn, batch_mean3, batch_var3, beta3, scale3, epsilon)
    conv3 = tf.nn.relu(bn3)
    pool_2 = max_pool_2x2(conv3)
    if SHOULD_PRINT_SHAPES and TRAIN_MODE:
        print("conv3 shape: {}".format(conv3.get_shape()))
        print("pool_2 shape: {}".format(pool_2.get_shape()))

    # third layer - conv
    w4 = weight_variable([2, 2, SECOND_CONV_TO_THIRD_CONV_MAPS, THIRD_CONV_TO_FOURTH_CONV_MAPS])
    z4_bn = conv2d(pool_2, w4)
    batch_mean4, batch_var4 = tf.nn.moments(z4_bn, [0])
    scale4 = tf.Variable(tf.ones([THIRD_CONV_TO_FOURTH_CONV_MAPS]), name="scale4")
    beta4 = tf.Variable(tf.zeros([THIRD_CONV_TO_FOURTH_CONV_MAPS]), name="beta4")
    bn4 = tf.nn.batch_normalization(z4_bn, batch_mean4, batch_var4, beta4, scale4, epsilon)
    conv4 = tf.nn.relu(bn4)
    pool_3 = max_pool_2x2(conv4)
    drop_2 = tf.nn.dropout(pool_3, keep_prob2)
    shapedrop3 = drop_2.get_shape()
    flat_size = 1
    first = True
    for mydim in shapedrop3:
        if first is False:
            flat_size *= mydim.value
        first = False
    flat = tf.reshape(drop_2, [-1, flat_size])
    if SHOULD_PRINT_SHAPES and TRAIN_MODE:
        print("conv4 shape: {}".format(conv4.get_shape()))
        print("pool_3 shape: {}".format(pool_3.get_shape()))
        print("drop_2 shape: {}".format(drop_2.get_shape()))
        print("flat shape: {}".format(flat.get_shape()))

    softmax = tf.nn.softmax(tf.layers.dense(inputs=flat, units=_NUM_CLASSES))
    if SHOULD_PRINT_SHAPES and TRAIN_MODE:
        print("softmax shape: {}".format(softmax.get_shape()))
    y_pred_cls = tf.argmax(softmax, axis=1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))
    if ADAM_OPTIMIZER:
        optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)
    else:
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # PREDICTION AND ACCURACY CALCULATION
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob2


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


def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory + "cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urllib.urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory + "./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)


def get_data_set(name="train"):
    x_local = None
    y_local = None

    maybe_download_and_extract()

    folder_name = "cifar_10"

    f = open('./data_set/' + folder_name + '/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/' + folder_name + '/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f)
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
        f = open('./data_set/' + folder_name + '/test_batch', 'rb')
        datadict = pickle.load(f)
        f.close()

        x_local = datadict["data"]
        y_local = np.array(datadict['labels'])

        x_local = np.array(x_local, dtype=float) / 255.0
        x_local = x_local.reshape([-1, 3, 32, 32])
        x_local = x_local.transpose([0, 2, 3, 1])
        x_local = x_local.reshape(-1, 32 * 32 * 3)

    return x_local, dense_to_one_hot(y_local)


TRAIN_MODE = True  # to load model and test only the test set - set on FALSE
ADAM_OPTIMIZER = True  # what model will be selected
SHOULD_SAVE_MODELS = False

SHOULD_PRINT_SHAPES = True
SHOULD_PRINT_VARS_NAMES = True
SHOULD_PRINT_START_VARS = True
SHOULD_PRINT_END_VARS = True
_BATCH_SIZE = 196
_EPOCH = 500
_TRAIN_KEEP_PROB = 0.5
X_TO_FIRST_CONV_MAPS = 0
FIRST_CONV_TO_SECOND_CONV_MAPS = 30
SECOND_CONV_TO_THIRD_CONV_MAPS = 40
THIRD_CONV_TO_FOURTH_CONV_MAPS = 95
_IMAGE_SIZE = 32
_IMAGE_CHANNELS = 3
_NUM_CLASSES = 10
BASE_PATH = "./hw2Models/{}42350Params/model.ckpt"  # uncomment to save new models
if ADAM_OPTIMIZER:
    SAVED_MODELS_PATH = BASE_PATH.format("adamOptimizer")
    OPTIMIZER_STR = "AdamOptimizer"
else:
    SAVED_MODELS_PATH = BASE_PATH.format("GradientOptimizer")
    OPTIMIZER_STR = "GradientDescentOptimizer"

# MODEL_PARAMS = 42350
# MODEL_BEST_ACC_ADAM = 79.01
# BEST_EPOCH_ADAM = 499
# MODEL_BEST_ACC_GRAD = 78.9
# BEST_EPOCH_ADAM = 339

tf.reset_default_graph()
sess = tf.Session()

if TRAIN_MODE:
    train_x, train_y = get_data_set("train")
    copy_train_x = train_x
    copy_train_x = copy_train_x.reshape(len(copy_train_x), _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS)

test_x, test_y = get_data_set("test")

x, y, loss, optimizer, correct_prediction, accuracy, y_pred_cls, keep_prob2 = model()
global_accuracy = 0
best_epoch = 0
train_error_list = []
train_loss_list = []
test_error_list = []
test_loss_list = []

if TRAIN_MODE:
    BATCHSIZE = int(math.ceil(len(train_x) / _BATCH_SIZE))
    _STEPS_PRINT = (BATCHSIZE / 2 - 1)

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
total_parameters = 0
if TRAIN_MODE:
    if SHOULD_PRINT_VARS_NAMES:
        print ("---vars name and shapes---")
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
        print ("---done vars---")


def args_print(stage, duration=0):
    print("{} ----------------------".format(stage))
    print("epochs {}".format(_EPOCH))
    print("batchSize {}".format(_BATCH_SIZE))
    print("keepProb {}".format(_TRAIN_KEEP_PROB))
    print("train_x dims: {}".format(train_x.shape))
    print("USE_SCALED_IMGS {}".format(USE_SCALED_IMGS))
    print("USE_ROTATED_IMGS {}".format(USE_ROTATED_IMGS))
    print("USE_FLIPPED_IMGS {}".format(USE_FLIPPED_IMGS))
    print("total PARAM {:,}".format(total_parameters))
    # print("X_TO_FIRST_CONV_MAPS {}".format(X_TO_FIRST_CONV_MAPS))
    print("FIRST_CONV_TO_SECOND_CONV_MAPS {}".format(FIRST_CONV_TO_SECOND_CONV_MAPS))
    print("SECOND_CONV_TO_THIRD_CONV_MAPS {}".format(SECOND_CONV_TO_THIRD_CONV_MAPS))
    print("THIRD_CONV_TO_FOURTH_CONV_MAPS {}".format(THIRD_CONV_TO_FOURTH_CONV_MAPS))
    print("global_accuracy {}".format(global_accuracy))
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("duration(formatted HH:MM:SS): {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))


if SHOULD_PRINT_START_VARS and TRAIN_MODE:
    args_print("start")


def main():
    if TRAIN_MODE:
        print_graph('test', 'x', 'y', [], [])  # to see we will get graphs and not place holder at the end
        total_start_time = time()
        for i in range(_EPOCH):
            print("\nEpoch: {0}/{1}\n".format((i + 1), _EPOCH))
            train(i)

        duration = time() - total_start_time
        if SHOULD_PRINT_END_VARS:
            args_print("end", duration)

        msg = "optimizer is {}, best accuracy = {} achieved on epoch number {}"
        print(msg.format(OPTIMIZER_STR, global_accuracy, best_epoch))
        print_graph('error/epochs(train in red, test in green)', 'epochs', 'error', train_error_list, test_error_list)
        print_graph('loss/epochs(train in red, test in green)', 'epochs', 'loss', train_loss_list, test_loss_list)

    else:
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
        sess2.run(init)
        saver.restore(sess2, SAVED_MODELS_PATH)
        print("Model restored from file: %s" % SAVED_MODELS_PATH)
        print("Checking test_x and test_y on best model with {}".format(OPTIMIZER_STR))
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
    global train_loss_list
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    current_error = 0
    current_loss = 0
    for s in range(batch_size):
        batch_xs = train_x[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]
        batch_ys = train_y[s * _BATCH_SIZE: (s + 1) * _BATCH_SIZE]

        _, batch_loss, batch_acc = sess.run(
            [optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, keep_prob2: _TRAIN_KEEP_PROB})

        # original: if s % 10
        if s % _STEPS_PRINT == 0:
            msg = "step: {} , batch_acc = {} , batch loss = {}"
            print(msg.format(s, batch_acc, batch_loss))

        current_loss += batch_loss
        current_error += 1-batch_acc

    # print("loss avg for epoch {} is {}".format(epoch, current_loss / batch_size))
    train_loss_list.append(current_loss / batch_size)
    train_error_list.append(current_error / batch_size)
    test_and_save(epoch)


def test_and_save(epoch):
    global global_accuracy
    global test_error_list
    global test_loss_list
    global best_epoch
    global saver

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    counter = 0
    current_test_loss = 0
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j], batch_test_loss = sess.run(
            [y_pred_cls, loss],
            feed_dict={x: batch_xs, y: batch_ys, keep_prob2: 1}
        )
        i = j
        current_test_loss += batch_test_loss
        counter += 1
    correct = (np.argmax(test_y, axis=1) == predicted_class)
    test_error_list.append(1-correct.mean())
    test_loss_list.append(current_test_loss / counter)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()

    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{}) - global best acc on epoch {} = {}"
    print(mes.format((epoch + 1), acc, correct_numbers, len(test_x), best_epoch, global_accuracy))

    if global_accuracy != 0 and global_accuracy < acc:
        mes = "epoch {} receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(epoch+1, acc, global_accuracy))
        if SHOULD_SAVE_MODELS:
            save_path = saver.save(sess, SAVED_MODELS_PATH)
            print("Model saved in file: %s" % save_path)
        global_accuracy = acc
        best_epoch = epoch + 1

    elif global_accuracy == 0:
        if SHOULD_SAVE_MODELS:
            save_path = saver.save(sess, SAVED_MODELS_PATH)
            print("Model saved in file: %s" % save_path)
        global_accuracy = acc
        best_epoch = epoch + 1

    print("###########################################################################################################")


if __name__ == "__main__":
    main()
