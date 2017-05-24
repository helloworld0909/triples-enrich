from collections import defaultdict
import numpy as np
import random


class HMMFactory(object):

    def __init__(self, filepath):
        self.input_file = open(filepath, 'r', encoding='utf-8').readlines()
        self.index2word = dict(enumerate(self.wordcount().keys()))
        self.word2index = dict((v, index) for index, v in self.index2word.items())
        self.n_word = len(self.index2word)

        # Deal with those characters did not appear in the training set
        self.index2word[self.n_word] = 'Unknown'
        self.word2index['Unknown'] = self.n_word
        self.n_word += 1

    def encode(self, raw_value):
        code = []
        for c in raw_value:
            if c in self.word2index:
                code.append(self.word2index[c])
            else:
                code.append(self.word2index['Unknown'])
        return code


    def wordcount(self):
        wordcount = defaultdict(int)
        for line in self.input_file:
            raw_value = line.strip().split('\t')[0]
            for c in raw_value:
                wordcount[c] += 1
        return wordcount

    def hiddenProb(self):
        probVector = [0.0]*2
        for line in self.input_file:
            split_value = line.strip().split('\t')[1]
            count = self.count_split(split_value)
            probVector[1] += count
            probVector[0] += len(split_value) - count
        total = sum(probVector)
        return np.array(probVector) / total


    @staticmethod
    def count_split(string):
        count = 0
        for c in string:
            if c == '|':
                count += 1
        return count

    def transMatrix(self):
        matrix = [[0.0, 0.0], [0.0, 0.0]]
        for line in self.input_file:
            raw_value, split_value = line.strip().split('\t')

            n_split = self.count_split(split_value)
            matrix[0][1] += n_split
            matrix[1][0] += n_split
            matrix[0][0] += len(split_value) - 2*n_split - 1
        transMatrix = np.empty(shape=(2, 2))
        for i in range(2):
            for j in range(2):
                transMatrix[i][j] = matrix[i][j] / sum(matrix[i])
        return transMatrix

    @staticmethod
    def toBoolean(split_value):
        ret = []
        for c in split_value:
            if c == '|':
                ret.append(1)
            else:
                ret.append(0)
        return ret

    def emissionMatrix(self):
        matrix = [[0.0]*self.n_word, [0.0]*self.n_word]
        for line in self.input_file:
            raw_value, split_value = line.strip().split('\t')

            if len(split_value) != len(raw_value):
                continue

            bool_split = self.toBoolean(split_value)
            for c0, c1 in zip(raw_value, bool_split):
                matrix[c1][self.word2index[c0]] += 1
        emissionMatrix = np.empty(shape=(2, self.n_word))
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                emissionMatrix[i][j] = matrix[i][j] / sum(matrix[i])
        return emissionMatrix


def encode_state(split_value):
    return list(map(lambda c:int(c == '|'), split_value))

def transform_state(state_list, raw_value):
    ret = []
    for state, c in zip(state_list, raw_value):
        if state == 1:
            ret.append('|')
        else:
            ret.append(c)
    return ''.join(ret)

def sample_file(filepath, n=100):
    with open(filepath, 'r', encoding='utf-8') as input_file:
        input_lines = input_file.readlines()
        sample = random.sample(input_lines, n)
        with open('test.txt', 'w', encoding='utf-8') as output_file:
            for line in sample:
                output_file.write(line)
        with open('train_.txt', 'w', encoding='utf-8') as new_input:
            for line in input_lines:
                if line not in sample:
                    new_input.write(line)

if __name__ == '__main__':
    sample_file('train.txt', n = 300)
