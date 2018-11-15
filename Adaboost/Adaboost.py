import pandas as pd
import re


def load_data():
    names = ['Sequence-name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Class']
    data = pd.read_csv('Ecoli.data.csv', names=names)
    print(data[:15])
    return data


def loadCsv(filename):
    file = open(filename, "rt")
    splitter = re.compile('\s+')
    new_file = filename + '.csv'
    csv_file = open(new_file, 'w+')
    lines = file.readlines()

    for line in lines:
        words = splitter.split(line.strip())
        for word in words[:-1]:
            csv_file.write(word + ',')
        csv_file.write(words[-1])
        csv_file.write('\r\n')

def main():
    loadCsv('Ecoli.data')
    # load_data()


main()
