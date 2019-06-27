import codecs
import csv


def count_csv(csv_data):
#    manifest_jsons = read_manifest(manifest_path)
    n = 0
    set_py = set()
    for line in csv_data:
        n += 1
        a = line[2].split(' ')
        for char in a:
            set_py.add(char)
        if n % 10000 == 0:
            print (n)
    return set_py


def main():
    
    data_files = ['data/train_py_tone.csv', 'data/test_py_tone.csv', 'data/dev_py_tone.csv']
    csv_data = []
    n = 0
    for data_file in data_files:
        with open(data_file, "rt", encoding="utf-8") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                csv_data.append(row)
                n += 1
                if n % 100000 == 0:
                    print (n)
    print (len(csv_data))
    
    set_py = count_csv(csv_data)
    with codecs.open('data/vocab_py_tone.txt', 'w', 'utf-8') as fout:
        for char in set_py:
            fout.write(char + '\n')


if __name__ == '__main__':
    main()
