import numpy as np
import re

from offlineEnrich import INFOBOX_ENRICH
from hmm.hmm import buildHMM
import hmm.util

menu_path = 'hmm/'

def base_main():
    test_file = open(menu_path + 'test.txt', 'r', encoding='utf-8')
    test_file_lines = test_file.readlines()
    TEST_NUM = len(test_file_lines)

    punctuation_pattern = re.compile(r'[/、,，;；|]')

    correct = 0
    for line in test_file_lines:
        raw_value, split_value = line.strip().split('\t')
        predict = re.sub(punctuation_pattern, '|', raw_value)

        if predict == split_value:
            correct += 1

    print('Base:', correct / TEST_NUM)

    test_file.close()

def mentioin_main():
    model = INFOBOX_ENRICH()
    test_file = open(menu_path + 'test.txt', 'r', encoding='utf-8')
    test_file_lines = test_file.readlines()
    TEST_NUM = len(test_file_lines)

    correct = 0
    for line in test_file_lines:
        raw_value, split_value = line.strip().split('\t')
        Y = split_value.strip().split('|')
        predict = model.enrich_infobox_value_segment(raw_value)

        if set(predict) == set(Y):
            correct += 1

    print('Mention:', correct / TEST_NUM)

    test_file.close()

def hmm_main():

    HMMFactory = hmm.util.HMMFactory(menu_path + 'train.txt')
    model = buildHMM(HMMFactory)

    diff_file = open(menu_path + 'diff.txt', 'w', encoding='utf-8')
    test_file = open(menu_path + 'test.txt', 'r', encoding='utf-8')
    test_file_lines = test_file.readlines()
    TEST_NUM = len(test_file_lines)

    correct = 0
    for line in test_file_lines:
        raw_value, split_value = line.strip().split('\t')

        Y = hmm.util.encode_state(split_value)
        X = np.atleast_2d(HMMFactory.encode(raw_value)).T
        score, predict = model.decode(X, algorithm='viterbi')
        if list(predict) == Y:
            correct += 1
        else:
            diff_file.write(raw_value + '\t' + split_value + '\t' + hmm.util.transform_state(predict, raw_value) + '\n')

    print('HMM:', correct / TEST_NUM)


    diff_file.close()
    test_file.close()

if __name__ == '__main__':
    base_main()
    mentioin_main()
    hmm_main()