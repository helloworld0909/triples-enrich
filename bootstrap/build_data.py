import json
import re
import offlineEnrich


def split_indices(mention_list):
    indices = []
    idx = 0
    for mention in mention_list:
        indices.append(idx + len(mention))
        idx += len(mention) + 1
    indices.pop(len(indices) - 1)
    return indices


def main():
    property_label = {}
    with open('../classifier/property.txt', 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split('\t')
            p = line[0]
            label = line[-1]
            property_label[p] = label

    with open('value_groupby_property.json', 'r', encoding='utf-8') as fin:
        jsonObj = json.load(fin)

    model = offlineEnrich.INFOBOX_ENRICH()
    punctuation_pattern = re.compile(r'[/、,，;；|]')

    transform = {}
    transform_test = {}
    count = 0
    for attr, values in jsonObj.items():
        if count % 100 == 0:
            print(count)
        body = {'isMulti': property_label[attr]}
        value_list = {}
        for value in values:
            if punctuation_pattern.search(value) is None:
                continue
            mention_list = model.split_one_value(attr, value, threshold=-1)
            indices = split_indices(mention_list)
            value_list[value] = indices
        body['values'] = value_list
        if hash(attr) % 10 < 9:
            transform[attr] = body
        else:
            transform_test[attr] = body
        count += 1

    with open("iter_0.json", 'w', encoding='utf-8') as fout:
        json.dump(transform, fout, indent=1, ensure_ascii=False, sort_keys=True)
    with open("test_data.json", 'w', encoding='utf-8') as fout:
        json.dump(transform_test, fout, indent=1, ensure_ascii=False, sort_keys=True)
    transform.update(transform_test)
    with open("all_data.json", 'w', encoding='utf-8') as fout:
        json.dump(transform, fout, indent=1, ensure_ascii=False, sort_keys=True)


if __name__ == '__main__':
    main()

