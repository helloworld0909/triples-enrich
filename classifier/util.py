from collections import defaultdict
import jieba

def encode_char(filename, threshold=0):
    char_dict = {}
    char_freq = defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            name = line.strip().split('\t')[0]
            for char in name:
                char_freq[char] += 1
    for char, freq in char_freq.items():
        if freq >= threshold:
            char_dict[char] = len(char_dict)

    return char_dict

def encode_word(filename, threshold=0):
    word_dict = {}
    word_freq = defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            name = line.strip().split('\t')[0]
            for word in jieba.cut(name):
                word_freq[word] += 1
    for word, freq in word_freq.items():
        if freq >= threshold:
            word_dict[word] = len(word_dict)

    return word_dict
