from xpinyin import Pinyin
import csv

p = Pinyin()

file_name = 'data/small_test_2.csv'
csv_file = csv.reader(open(file_name,'r'))
out = open("data/test_pinyin.csv", "a", newline = "")
csv_writer = csv.writer(out, dialect = "excel")

for line in csv_file:
    py = p.get_pinyin(line[2],' ')
    row = [line[0],line[1],py]
    csv_writer.writerow(row)
