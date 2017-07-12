import os
import numpy as np
import re

from offlineEnrich import INFOBOX_ENRICH
from hmm.hmm import buildHMM
import hmm.util
from crf.CRFMetric import CRFMetric

menu_path = 'E:\\python_workspace\\enrich\\output\\'
punctuation_pattern = re.compile(r'[/、,，;；|]')

def base_main():
    global punctuation_pattern

    test_file = open(menu_path + 'test_labeled.txt', 'r', encoding='utf-8')

    count = 0
    correct = 0
    for line in test_file:
        raw_value, split_value = line.strip().split('\t')
        predict_value = re.sub(punctuation_pattern, '|', raw_value)

        for raw, label, predict in zip(raw_value, split_value, predict_value):
            if re.search(punctuation_pattern, raw) is not None:
                count += 1
                if label == predict:
                    correct += 1

    print('Base:', count, correct, correct / float(count))

    test_file.close()

def mentioin_main():
    global punctuation_pattern

    model = INFOBOX_ENRICH()
    test_file = open(menu_path + 'test_labeled.txt', 'r', encoding='utf-8')

    count = 0
    correct = 0
    error_count = 0
    for line in test_file:
        raw_value, split_value = line.strip().split('\t')

        predict = model.enrich_infobox_value_segment(raw_value)
        predict_value = '|'.join(predict)

        # 和offlineEnrich的操作保持一致，先排序
        split_value_sorted = sorted(list(set(split_value.strip().split('|'))))
        split_value = '|'.join(split_value_sorted)

        # print(split_value, predict_value)

        if len(predict_value) != len(split_value) or len(raw_value) != len(split_value):
            error_count += 1
            # print(split_value, predict_value)
            continue
        else:
            for raw, label, predict in zip(raw_value, split_value, predict_value):
                if re.search(punctuation_pattern, raw) is not None:
                    count += 1
                    if label == predict:
                        correct += 1

    print('Mention:', count, correct, correct / float(count))
    # print('Length not match count:', error_count)

    test_file.close()

def hmm_main():

    HMMFactory = hmm.util.HMMFactory(menu_path + 'train_labeled.txt')
    model = buildHMM(HMMFactory)

    # diff_file = open(menu_path + 'diff.txt', 'w', encoding='utf-8')
    test_file = open(menu_path + 'test_labeled.txt', 'r', encoding='utf-8')

    count = 0
    correct = 0
    error_count = 0
    for line in test_file:
        raw_value, split_value = line.strip().split('\t')

        Y = hmm.util.encode_state(split_value)

        X = np.atleast_2d(HMMFactory.encode(raw_value)).T
        score, predict = model.decode(X, algorithm='viterbi')

        predict_encode = list(predict)

        if len(predict_encode) != len(Y) or len(raw_value) != len(Y):
            error_count += 1
            print(split_value)
            continue
        else:
            for raw, label, predict in zip(raw_value, Y, predict_encode):
                if re.search(punctuation_pattern, raw) is not None:
                    count += 1
                    if label == predict:
                        correct += 1

    print('HMM:', count, correct, correct / float(count))

    test_file.close()

def crf_main():
    global punctuation_pattern

    crf_path = 'crf\\CRF++-0.54\\'
    crf_data_path = crf_path + 'example\\valueSeg\\'
    os.system('{}crf_learn -a MIRA {}template {}train.data model'.format(crf_path, crf_data_path, crf_data_path))
    os.system('{}crf_test -m model {}test.data > crf\\output.txt'.format(crf_path, crf_data_path))

    metric = CRFMetric()
    count, correct = metric.punctuation_accuracy('crf\\output.txt', punctuation_pattern)
    print('CRF:', count, correct, correct / float(count))

if __name__ == '__main__':
    base_main()
    mentioin_main()
    hmm_main()
    crf_main()