"""Build vocabulary from manifest files.

Each item in vocabulary file is a character.
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import argparse
#import functools
import codecs
#import json
from collections import Counter
#import os.path
#import _init_paths
import csv


def count_csv(counter, csv_data):
#    manifest_jsons = read_manifest(manifest_path)
    count = 0
    for line in csv_data:
        count += 1
        for char in line[2]:
            counter.update(char)
        if count % 10000 == 0:
            print (count)


def main():
    count_threshold = 10
    data_files = ['ST-CMDS-20170001_1-OS/train.csv', 'ST-CMDS-20170001_1-OS/test.csv', 'ST-CMDS-20170001_1-OS/dev.csv']
    csv_data = []
    count = 0
    for data_file in data_files:
        with open(data_file, "rt", encoding="utf-8") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                csv_data.append(row)
                count += 1
                if count % 100000 == 0:
                    print (count)
    print (len(csv_data))
    counter = Counter()
    #csv_data = csv_data[:100]
    count_csv(counter, csv_data)
    print (counter)
    print (len(counter))
    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=False)
    with codecs.open('vocab_4500h_all_data_reverse_greater_10.txt', 'w', 'utf-8') as fout:
        for char, count in count_sorted:
            if count < count_threshold: continue
            fout.write(char + '\n')


if __name__ == '__main__':
    main()
