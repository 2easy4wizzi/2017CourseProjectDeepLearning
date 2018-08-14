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

logging.getLogger().setLevel(logging.INFO)

LINE_FROM_CLASS = 5000
MINIMUM_ROW_LENGTH = 25
MAXIMUM_ROW_LENGTH = 150

# GENERAL VARS
PRO_FLD = ''
DATA_DIR = 'input/'
EMB_FILE = 'glove.6B.50d'
EMB_FILE_PATH = PRO_FLD + DATA_DIR + EMB_FILE
DATA_FILE = '2way_rus_usa{}-{}'.format(MINIMUM_ROW_LENGTH, MAXIMUM_ROW_LENGTH)
DATA_FILE_PATH = PRO_FLD + DATA_DIR + DATA_FILE + '.txt'
COUNT_WORD = 20  # if a sentence has COUNT_WORD of the same word - it's a bad sentence (just a troll)

# existing FILES
TRA_FLD = 'trained_results_1533109035/'  # NOT used if USE_TMP_FOLDER is TRUE !!!
USE_TMP_FOLDER = True
PRINT_CLASSES_STATS_EACH_X_STEPS = 1  # prints dev stats each x steps


def clean_str(s):  # DATA
    strip_special_chars = re.compile("[^A-Za-z0-9 ,.]+")
    s = s.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", s)


def convert_data_to_indices_of_emb_mat(x_data):

    return


def load_data(data_full_path):
    l_train_x, l_train_y, l_dev_x, l_dev_y, l_test_x, l_test_y = [], [], [], [], [], []
    return l_train_x, l_train_y, l_dev_x, l_dev_y, l_test_x, l_test_y


def load_emb(emb_full_path):
    l_word_to_emb_mat_ind, l_emb_mat = [], []
    return l_word_to_emb_mat_ind, l_emb_mat


def get_bidirectional_rnn_model():

    return


def train():

    return


def test():

    return


def args_print(stage, duration=0):
    print("{} ----------------------".format(stage))
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time(HH:MM:SS): {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))
    return


if __name__ == '__main__':
    print("Entering function __main__")
    total_start_time = time.time()
    word_to_emb_mat_ind, emb_mat = load_emb(EMB_FILE_PATH)
    train_x, train_y, dev_x, dev_y, test_x, test_y = load_data(DATA_FILE_PATH)
    get_bidirectional_rnn_model()
    train()
    test()
    dur = time.time() - total_start_time
    args_print('End summary', int(dur))
    print("Leaving function __main__")

