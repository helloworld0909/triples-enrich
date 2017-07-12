import re

class CRFMetric(object):
    def __init__(self):
        pass

    @staticmethod
    def sentence_accuracy(filepath):
        correct = 0
        count = 0

        with open(filepath, 'r', encoding='utf-8') as input_file:
            match = True
            for line in input_file:
                if line.strip() == '':
                    count += 1
                    correct += int(match)
                    match = True
                else:
                    if not match:
                        continue
                    else:
                        label, predict = line.strip().split('\t')[-2:]
                        if label != predict:
                            match = False
        return correct / float(count)

    @staticmethod
    def punctuation_accuracy(filepath, punctuation_pattern):
        correct = 0
        count = 0

        with open(filepath, 'r', encoding='utf-8') as input_file:

            for line in input_file:
                if line.strip() == '':
                    continue

                data_tuple = line.strip().split('\t')
                c = data_tuple[0]

                if re.search(punctuation_pattern, c) is not None:
                    count += 1
                    label, predict = data_tuple[-2:]
                    if label == predict:
                        correct += 1
        return count, correct


