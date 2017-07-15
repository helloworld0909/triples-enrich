import jieba.posseg as posseg
import logging
import re

logging.basicConfig(level=logging.INFO)

class CRFFactory(object):

    menu_path = "E:\\python_workspace\\enrich\\"
    mention_path = menu_path + "middleware\\mention_list.txt"
    multivalued_path = menu_path + "middleware\\multivalued_attribute_list.txt"

    def __init__(self):
        self.mention_set = set()
        self.multivalued_set = set()
        self.load_mention_set()
        self.load_multivalued_set()
        self.punctuation_pattern = re.compile(r'[/、,，;；|]')

    def load_mention_set(self):

        with open(self.mention_path, 'r', encoding = "utf-8") as input_file:
            for line in input_file:
                line = line.strip()
                self.mention_set.add(line)

    def load_multivalued_set(self):
        with open(self.multivalued_path, 'r', encoding="utf-8") as input_file:
            for line in input_file:
                line = line.strip()
                self.multivalued_set.add(line)

    def build(self, filepath, output='train.data'):
        error_count = 0
        with open(filepath, 'r', encoding='utf-8') as input_file:
            with open(output, 'w', encoding='utf-8') as output_file:
                for line in input_file:
                    attribute, before_segment, after_segment = line.strip().split('\t')

                    if len(before_segment) != len(after_segment):
                        error_count += 1
                        continue

                    mention_tag = self.get_mention_tag(before_segment)
                    word_segment_tag, word_nominal_tag = self.get_word_segment_nominal_tag(before_segment)
                    segment_tag = self.get_segment_tag(after_segment)
                    meta_tag = self.get_meta_tag(attribute, before_segment)

                    # CRF++不把空格当做字符，需要替换掉
                    before_segment = before_segment.replace(' ', '□')
                    all_tags = [
                        before_segment,
                        mention_tag,
                        word_segment_tag,
                        word_nominal_tag,
                        meta_tag,
                        segment_tag
                    ]

                    for i in range(len(before_segment)):
                        tags = map(lambda l:l[i], all_tags)
                        output_file.write('\t'.join(tags) + '\n')

                    output_file.write('\n')
        logging.info('Finished. Found {} error lines'.format(error_count))


    def get_mention_tag(self, value):
        length = len(value)

        punctuations = re.findall(self.punctuation_pattern, value)

        if punctuations is None:
            return ['MO'] * length

        mention_tag_candidates = []
        for punctuation in set(punctuations):
            word_list = value.split(punctuation)
            logging.debug(word_list)

            tag_list, num_mention = self.transform_mention(word_list)
            logging.debug(tag_list)

            mention_tag = sum(self.join_mention(tag_list), [])  #Join tag_list with 'MO', and then flatten it

            score = 2 * num_mention - len(tag_list)
            mention_tag_candidates.append((mention_tag, score))

        most_likely_mention_tag = max(mention_tag_candidates, key=lambda c:c[1])[0]    #取score最高的那一种分法作为标注
        assert len(most_likely_mention_tag) == length, 'Mention_tag error: ' + value + '\t' + str(most_likely_mention_tag)

        return most_likely_mention_tag

    def transform_mention(self, word_list):
        tag_list = []
        num_mention = 0
        for word in word_list:
            if not word:
                tag_list.append([])
            elif word.strip() in self.mention_set:    #去两端空格再找mention
                tag_list.append(['MB'] + ['MI'] * (len(word) - 1))
                num_mention += 1
            else:
                tag_list.append(['MO'] * len(word))
        return tag_list, num_mention

    @staticmethod
    def join_mention(tag_list):
        for index in range(1, 2 * len(tag_list) - 1, 2):
            tag_list.insert(index, ['MO'])
        return tag_list

    def get_word_segment_nominal_tag(self, value):
        word_segment_tag = []
        word_nominal_tag = []

        for w in posseg.cut(value):
            tag_list = self.get_word_segment_tag(w.word)
            word_segment_tag.extend(tag_list)
            word_nominal_tag.extend(self.get_flag(tag_list, w.flag))

        assert len(word_segment_tag) == len(value), 'Segment_tag error: ' + value
        assert len(word_nominal_tag) == len(value), 'Nominal_tag error: ' + value

        return word_segment_tag, word_nominal_tag


    @staticmethod
    def get_word_segment_tag(word):
        length = len(word)
        if length == 0:
            raise Exception('Length of word is zero')
        elif length == 1:
            return ['S']
        else:
            tag_list = ['B']
            for _ in range(length - 2):
                tag_list.append('M')
            tag_list.append('E')
            return tag_list

    @staticmethod
    def get_flag(tag_list, flag):
        return [flag] * len(tag_list)

    @staticmethod
    def add_flag(tag_list, flag):
        return list(map(lambda t: t + '-' + flag, tag_list))

    @staticmethod
    def get_segment_tag(after_segment):
        segment_tag = []
        for c in after_segment:
            if c == '|':
                segment_tag.append('B')
            else:
                segment_tag.append('O')
        return segment_tag

    def get_meta_tag(self, attribute, value):
        if attribute in self.multivalued_set:
            return ['I'] * len(value)
        else:
            return ['O'] * len(value)

if __name__ == '__main__':

    crfFactory = CRFFactory()
    crfFactory.build('E:\\python_workspace\\enrich\\input\\train_labeled_with_attribute.txt', output='train.data')
    crfFactory.build('E:\\python_workspace\\enrich\\input\\test_labeled_with_attribute.txt', output='test.data')