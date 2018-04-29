import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences

maxAttrLen = 10
maxValueLen = 30


def init_word2idx(filename='all_data.json'):
    with open(filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    attr_word2idx = {'PADDING_TOKEN': 0}
    value_word2idx = {'PADDING_TOKEN': 0}

    for attr, body in data.items():
        for c in attr:
            if c not in attr_word2idx:
                attr_word2idx[c] = len(attr_word2idx)
        for value in body['values'].keys():
            for c in value:
                if c not in value_word2idx:
                    value_word2idx[c] = len(value_word2idx)

    return attr_word2idx, value_word2idx


def init_sequences(filename, attr_word2idx, value_word2idx):
    with open(filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    attr_sequences = []
    attr_labels = []
    value_sequences = []
    value_tag_sequences = []

    attr_label_lookup = {
        "0": 0,
        "1": 1
    }

    for attr, body in data.items():
        attr_seq = list(map(lambda c: attr_word2idx.get(c, 0), attr))
        attr_label = np.zeros(len(attr_label_lookup))
        attr_label[attr_label_lookup[body['isMulti']]] = 1

        for value, indices in body['values'].items():
            value_seq = list(map(lambda c: value_word2idx.get(c, 0), value))
            value_tag_seq = [1] * len(value_seq)
            for idx in indices:
                value_tag_seq[idx] = 2

            attr_sequences.append(attr_seq)
            attr_labels.append(attr_label)
            value_sequences.append(value_seq)
            value_tag_sequences.append(value_tag_seq)

    attr_sequences = pad_sequences(attr_sequences, maxlen=maxAttrLen)
    attr_labels = np.array(attr_labels)
    value_sequences = pad_sequences(value_sequences, maxlen=maxValueLen)
    value_tag_sequences = np.expand_dims(pad_sequences(value_tag_sequences, maxlen=maxValueLen), axis=-1)

    return attr_sequences, value_sequences, attr_labels, value_tag_sequences


if __name__ == '__main__':
    attr_word2idx, value_word2idx = init_word2idx()
    print(init_sequences('iter0.json', attr_word2idx, value_word2idx))
