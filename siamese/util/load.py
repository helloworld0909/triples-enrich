import os
import json
import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences

DATA_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../bootstrap/data/"))

PUNCTUATION_LIST = ['/', '、', ',', '，', ';', '；', '|']

maxAttrLen = 10
maxValueLen = 20

random.seed(0)


def init_word2idx(filename='all_data.json'):
    with open(os.path.join(DATA_PATH, filename), 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    word2idx = {'PADDING_TOKEN': 0, 'UNKNOWN_TOKEN': 1}

    for attr, body in data.items():
        for c in attr:
            if c not in word2idx:
                word2idx[c] = len(word2idx)
        for value in body['values'].keys():
            for c in value:
                if c not in word2idx:
                    word2idx[c] = len(word2idx)

    return word2idx


def load_labels(filename, word2idx):
    with open(os.path.join(DATA_PATH, filename), 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    attr_sequences = []
    value_sequences = []
    y = []

    for attr, body in data.items():
        attr_seq = list(map(lambda c: word2idx.get(c, 1), attr))

        for value, indices in body['values'].items():
            value_seq = list(map(lambda c: word2idx.get(c, 1), value))

            attr_sequences.append(attr_seq)
            value_sequences.append(value_seq)
            y.append(1)

        if len(body['values']) < 4:
            continue

        samples = list(body['values'].keys())
        for _ in range(len(body['values'])):
            punc = PUNCTUATION_LIST[random.randint(0, len(PUNCTUATION_LIST) - 1)]
            false_value = punc.join(random.sample(samples, random.randint(2, 4)))
            false_seq = list(map(lambda c: word2idx.get(c, 1), false_value))

            attr_sequences.append(attr_seq)
            value_sequences.append(false_seq)
            y.append(0)

    attr_sequences = pad_sequences(attr_sequences, maxlen=maxAttrLen)
    value_sequences = pad_sequences(value_sequences, maxlen=maxValueLen)
    y = np.array(y)

    return attr_sequences, value_sequences, y


if __name__ == '__main__':
    lookup = init_word2idx()
    print(load_labels('all_data.json', lookup))
