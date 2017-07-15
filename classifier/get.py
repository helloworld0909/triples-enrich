import re


lookup = {}
with open('../infobox_testdata_enrich1.txt', 'r', encoding='utf-8') as input_file:
    for line in input_file:
        try:
            _, p, o = line.strip().split('\t')
            o = o.replace('<a>', '')
            o = o.replace('</a>', '')
            lookup[o] = p
        except:
            pass

with open('../output/train_labeled.txt', 'r', encoding='utf-8') as input_file:
    output_file = open('../train_labeled_with_attribute.txt', 'w', encoding='utf-8')
    for line in input_file:
        o = line.strip().split('\t')[0]
        if o in lookup:
            p = lookup[o]
            output_file.write(p + '\t' + line)
        else:
            if o + '等' in lookup:
                p = lookup[o + '等']
                output_file.write(p + '\t' + line)
            else:
                print(line)


