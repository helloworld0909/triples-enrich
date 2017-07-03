from collections import defaultdict
import time

infobox_path = 'enrich_triples.txt'
stat_property = defaultdict(lambda : [0, 0])   #第一个代表属性出现次数，第二个表示这个属性的值被分割的次数

def stat_output():
    with open('stat.txt', 'w', encoding='utf-8') as stat_file:
        for p, stats in sorted(stat_property.items(), key=lambda a:a[1][0], reverse=True):
            stat_file.write(p + '\t' + str(stats[0]) + '\t' + str(stats[1]) + '\n')

if __name__ == '__main__':

    with open(infobox_path, 'r', encoding='utf-8') as input_file:
        last_s = ''
        last_p = ''
        counted = False

        count = 0
        for line in input_file:
            count += 1
            if count % 100000 == 0:
                print(count, time.ctime())
            try:
                s, p, o = line.strip().split('\t')
                if s == last_s:
                    if p == last_p and not counted:
                        stat_property[p][1] += 1
                        counted = True
                    if p == last_p and counted:
                        pass
                    else:
                        stat_property[p][0] += 1
                        counted = False
                else:
                    stat_property[p][0] += 1
                    counted = False

                last_s = s
                last_p = p
            except Exception as e:
                print(line, e)

    stat_output()
