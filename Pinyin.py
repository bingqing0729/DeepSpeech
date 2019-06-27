from xpinyin import Pinyin
import csv

p = Pinyin()


file_name = 'data/dev_clean.csv'
csv_file = csv.reader(open(file_name,'r'))
out = open("data/dev_py_tone.csv", "a", newline = "")
csv_writer = csv.writer(out, dialect = "excel")

for line in csv_file:
    py = p.get_pinyin(line[2],' ', tone_marks='numbers')
    row = [line[0],line[1],py]
    csv_writer.writerow(row)

'''
file_name = 'ifeng_corpus.txt'
out = open('ifeng_pinyin.txt','w')

for line in open(file_name,"r"):
    py = p.get_pinyin(line[:-1],'')
    print(py+'\n')
    out.write(py+'\n')

out.close()
'''
