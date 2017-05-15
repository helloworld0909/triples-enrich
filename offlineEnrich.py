#-*- coding:UTF-8 -*-
'''
Created on 2017-4-27

@author: Bo Xu <mailto:bolang1988@gmail.com>

@version: 1.0

@summary:

'''

import re
from convert2Date import CONVERT_DATE
import time

class INFOBOX_ENRICH:


    menu_path = ""
    attribute_match_path = menu_path + "middleware\\similar_attribute_pair_human.txt"
    date_attribute_path = menu_path + "middleware\\date_attribute_list.txt"
    mention_path = menu_path + "middleware\\mention_list.txt"

    mention_2_attribute_dict = None
    date_attribute_set = None
    mention_set = None
    punctuation_list = [' ', '/', '、', ',', '，', ';', '；', '|']


    def __init__(self):
        self.load_mention_2_attribute_dict()
        self.load_date_attribute_set()
        self.load_mention_set()




    def load_mention_2_attribute_dict(self):
        self.mention_2_attribute_dict = dict()
        with open(self.attribute_match_path, 'r', encoding = "utf-8") as f1:
            for line1 in f1:
                line1 = line1.rstrip()
                words1 = line1.split("\t")
                attribute1 = words1[0]
                attribute_list1 = words1[1].split("|||")
                for attribute2 in attribute_list1:
                    if attribute2 not in self.mention_2_attribute_dict:
                        self.mention_2_attribute_dict[attribute2] = attribute1
        f1.close()

    def load_date_attribute_set(self):
        self.date_attribute_set = set()
        with open(self.date_attribute_path, 'r', encoding = "utf-8") as f1:
            for line1 in f1:
                line1 = line1.rstrip()
                self.date_attribute_set.add(line1)
        f1.close()

    def load_mention_set(self):
        self.mention_set = set()
        with open(self.mention_path, 'r', encoding = "utf-8") as f1:
            for line1 in f1:
                line1 = line1.rstrip()
                self.mention_set.add(line1)
        f1.close()

    def enrich_infobox_attribute_normalize(self, p):
        if p not in self.mention_2_attribute_dict:
            return p
        else:
            norm_p = self.mention_2_attribute_dict[p]
            return norm_p

    def enrich_infobox_date_normalize(self, p, o):
        if p not in self.date_attribute_set:
            return o
        else:
            newo = CONVERT_DATE().run(o)
            if newo == None:
                newo = o

            return newo


    def replace_olddata_label(self, z):
        z = str(z).strip().replace('\xa0', '').replace('&nbsp;', ' ')
        z = re.sub('[\r\n\t]', '', z)
        z = re.sub('<sup>.+?</sup>', '', z)
        z = z.replace('<a>', '*a*').replace('</a>', '*/a*')
        z = re.sub('<br/?>', '|||', z)
        z = re.sub('<.+?>', '', z)
        z = z.replace('*a*', '<a>').replace('*/a*', '</a>').strip()
        z = z.replace('<a></a>', '').strip()
        return z

    def value_preprocess(self, value_string):
        pattern1 = re.compile(r'<a[^>]*>([^<]*)</a>')

        pattern3 = re.compile(r'^（([^（）]*)）$')
        pattern4 = re.compile(r'^《([^《》]*)》$')

        value_string = value_string.strip()
        value_string = value_string.replace("&quot;", '"').replace("&amp;", '&').replace("&lt;", '<').replace("&gt;", '>')
        value_string = re.sub(pattern1,r'\1', value_string)
        value_string = re.sub(pattern3,r'\1', value_string)
        value_string = re.sub(pattern4,r'\1', value_string)

        return value_string


    def check_mentions(self, word_list):
        score = 0
        mention_list = list()

        end_word = word_list[-1]


        if end_word.endswith("等"):
            if end_word not in self.mention_set and end_word[:-1] in self.mention_set:
                end_word = end_word[:-1]

        if end_word in self.mention_set:
            score +=1
        else:
            score -=1


        for word in word_list[:-1]:
            mention_list.append(word)

            if word in self.mention_set== True:
                score +=1
            else:
                score -=1

        mention_list.append(end_word)

        return score, mention_list



    def split_one_value(self, one_value):
        punctuation_pattern = re.compile(r'[/、,，;；|]')

        r1 = re.findall(punctuation_pattern, one_value)

        base_score = 1.1
        current_mention_list = [one_value]

        if r1 == None: return current_mention_list

        for split_word in set(r1):
            word_list = one_value.split(split_word)
            score, mention_list = self.check_mentions(word_list)
            if score > base_score:
                base_score = score
                current_mention_list = mention_list

        return current_mention_list

    def enrich_infobox_value_segment(self, o):
        split_o_list = list()

        o = self.value_preprocess(o)

        for split_o in o.split("|||"):
            value_list = self.split_one_value(split_o)
            if len(value_list) == 1:
                split_o_list.append(split_o)
                continue
            for v in value_list:
                if value_list in self.punctuation_list: continue
                split_o_list.append(v)

        split_o_list = sorted(list(set(split_o_list)))

        return split_o_list


    def process_one_pair(self, p, o):

        p1 = self.enrich_infobox_attribute_normalize(p)

        o2 = self.enrich_infobox_date_normalize(p1, o)

        o3_list = self.enrich_infobox_value_segment(o2)

        return p1, o3_list

    def run(self, infobox_path, enrich_infobox_path):
        enrich_infobox_file = open(enrich_infobox_path, 'w', encoding = "utf-8")

        count = 0
        with open(infobox_path, 'r', encoding = "utf-8") as f1:
            for line1 in f1:
                count += 1
                if count % 10000 == 0:
                    print(count, time.ctime())


                try:
                    line1 = line1.rstrip()
                    words1 = line1.split("\t")
                    s = words1[0]
                    p = words1[1]
                    o = words1[2]

                    if p == "CATEGORY_ZH" or p == "DESC":
                        enrich_infobox_file.write(s + "\t" + p + "\t" + o + "\n")
                        continue

                    o = self.replace_olddata_label(o)

                    newp, new_o_list = self.process_one_pair(p, o)

                    if newp == "" or newp == None: continue

                    for newo in new_o_list:
                        if newo == "" or newo == None: continue
                        if newo in self.mention_set: newo = "<a>" + newo + "</a>"
                        enrich_infobox_file.write(s + "\t" + newp + "\t" + newo + "\n")

                except:
                    print("error", line1)


if __name__ == "__main__":
    infobox_path = "infobox_testdata_enrich1.txt"
    enrich_infobox_path = "enrich_triples.txt"

    INFOBOX_ENRICH().run(infobox_path, enrich_infobox_path)
