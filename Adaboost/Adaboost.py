import pandas as pd
import re
import pandas
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier


def load_data():
    names = ['Sequence-name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Class']
    data = pd.read_csv('Ecoli.data.csv', names=names)
    # print(data[:15])

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


def begin_prediction(dataframe):
    encoder = preprocessing.OneHotEncoder()
    enc = preprocessing.LabelBinarizer()
    X1 = dataframe.iloc[:, 0]
    X = dataframe.iloc[:, 1:8]
    Y = dataframe.iloc[:, 8]
    seed = 7
    num_trees = 100
    kfold = model_selection.RepeatedKFold(n_splits=5, random_state=seed, n_repeats=10)

    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)

    for i in range(len(results)):
        print('Accuracy for fold %d is: %.2f%%' % ((i + 1), results[i] * 100))

    print('Mean accuracy: %.2f%%' % (results.mean() * 100))
    print('Standard Deviation: %.2f' % results.std())


def main():
    loadCsv('Ecoli.data')
    data = load_data()
    begin_prediction(data)


main()
