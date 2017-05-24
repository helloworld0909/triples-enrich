import numpy as np
from hmmlearn.hmm import MultinomialHMM
import util

def buildHMM(HMMFactory):

    model = MultinomialHMM(n_components=2, n_iter=200)
    model.startprob_ = HMMFactory.hiddenProb()
    model.transmat_ = HMMFactory.transMatrix()
    model.emissionprob_ = HMMFactory.emissionMatrix()
    return model

if __name__ == '__main__':

    HMMFactory = util.HMMFactory('train.txt')
    model = buildHMM(HMMFactory)

    ## Test
    # print(HMMFactory.hiddenProb())
    # print(HMMFactory.encode('莴苣，大葱，杏子'))
    # print(HMMFactory.encode_state('左启泽|张媛媛|吕国平|侯东星'))
    # print(HMMFactory.hiddenProb())
    # print(HMMFactory.transMatrix())
    # print(HMMFactory.emissionMatrix())
    # print(util.transform_state([0, 0, 1, 0, 0], '剧情，幻想'))

    diff_file = open('diff.txt', 'w', encoding='utf-8')
    test_file = open('test.txt', 'r', encoding='utf-8')
    test_file_lines = test_file.readlines()
    TEST_NUM = len(test_file_lines)

    correct = 0
    skip = 0
    for line in test_file_lines:
        raw_value, split_value = line.strip().split('\t')
        if len(raw_value) != len(split_value):
            skip += 1
            continue
        Y = HMMFactory.encode_state(split_value)
        X = np.atleast_2d(HMMFactory.encode(raw_value)).T
        score, predict = model.decode(X, algorithm='viterbi')
        if list(predict) == Y:
            correct += 1
        else:
            diff_file.write(raw_value + '\t' + split_value + '\t' + util.transform_state(predict, raw_value) + '\n')

    print(skip)
    print(correct / (TEST_NUM - skip))

    diff_file.close()
    test_file.close()
