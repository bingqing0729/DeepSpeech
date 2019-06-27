import re
import csv


csv_file = open('data/dev_clean.csv', 'w')
csv_write = csv.writer(csv_file, dialect='excel')
csv_write.writerow(['wav_filename', 'wav_filesize', 'transcript'])

vocab_file = open('data/vocab_4500h_all_data_reverse_greater_10.txt', 'r')
f = open('ST-CMDS-20170001_1-OS/dev.csv', 'r')
vocab_dict = {}
vocab_lines = vocab_file.readlines()
for each in vocab_lines:
    text = each.rstrip()
    vocab_dict[each.rstrip()] = 1
print (vocab_dict)
reader = csv.reader(f)

count = 0
valid_count = 0
for row in reader:
    count += 1
    text = row[2]
    valid = True
    for each in text:
        if vocab_dict.get(each, 0) == 0:
            valid = False
            break
    if valid:
        valid_count += 1
        csv_write.writerow([row[0], row[1], row[2]])
    if count % 10000 == 0:
        print (count, valid_count)
print (count, valid)
    #if count > 10:
    #    break
