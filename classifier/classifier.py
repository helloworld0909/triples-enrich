import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

from util import encode_char, encode_word

valid_split = 0.1
random.seed(0)

def load_data(filename):
    X = []
    Y = []

    with open(filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            name, _, split_count, split_rate, label = line.strip().split('\t')

            split_count = int(split_count)
            split_rate = float(split_rate)
            label = int(label)

            X.append(np.array((split_rate, split_count), dtype='float32'))
            Y.append(label)

    X = scale(X)

    return X, Y

def load_char_feature(filename, char_dict):
    X = []
    code_dim = len(char_dict) + 1   # For Unknown char
    with open(filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            code = np.zeros(code_dim, dtype='float32')
            name = line.strip().split('\t')[0]
            for char in name:
                code[char_dict.get(char, -1)] = 1
            X.append(code)

    return X

def LR(X, Y):

    lr = LogisticRegression()

    scores = cross_val_score(lr, X, Y, cv=10)

    # print(scores)
    print('AvgScore:', str(sum(scores) / len(scores)))


def rate_count_char():
    filepath = 'property.txt'

    char_dict = encode_char(filepath, threshold=4)
    X1, Y = load_data(filepath)
    X2 = load_char_feature(filepath,char_dict)

    X = []
    for x1, x2 in zip(X1, X2):
        X.append(np.concatenate((x1, x2)))

    LR(X, Y)

def rate_count_word():
    filepath = 'property.txt'

    word_dict = encode_word(filepath, threshold=0)
    X1, Y = load_data(filepath)
    X2 = load_char_feature(filepath, word_dict)

    X = []
    for x1, x2 in zip(X1, X2):
        X.append(np.concatenate((x1, x2)))

    LR(X, Y)

def rate_count():
    filepath = 'property.txt'

    X1, Y = load_data(filepath)

    LR(X1, Y)

def char():
    filepath = 'property.txt'

    char_dict = encode_char(filepath, threshold=4)
    _, Y = load_data(filepath)
    X2 = load_char_feature(filepath, char_dict)

    LR(X2, Y)

if __name__ == '__main__':
    print('rate_count:')
    rate_count()

    print('char:')
    char()

    print('rate_count_char')
    rate_count_char()

    print('rate_count_word')
    rate_count_word()