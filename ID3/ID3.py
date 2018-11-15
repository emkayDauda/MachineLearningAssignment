import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn import preprocessing
import statistics
from sklearn import model_selection


def load_data():
    names = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment',
             'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
             'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
             'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
             'spore-print-color', 'population', 'habitat', 'Class']

    data = pd.read_csv('Mushroom.data.csv', names=names)
    # print(data[:15])

    imp_mean = SimpleImputer(missing_values='?', strategy='most_frequent')
    data = imp_mean.fit_transform(data)
    data = pd.DataFrame(data=data, index=range(len(data)), columns=names)
    # print(data.head())
    return data


def begin_prediction(dataframe):
    encoder = preprocessing.OneHotEncoder()

    X = dataframe.iloc[:, 0:22]
    Y = dataframe.iloc[:, 22]
    seed = 3
    num_trees = 100
    max_features = 3
    kfold = model_selection.RepeatedKFold(n_splits=5, random_state=seed, n_repeats=10)
    model = tree.DecisionTreeClassifier()
    results = model_selection.cross_val_score(model, encoder.fit_transform(X), Y, cv=kfold)
    for i in range(len(results)):
        print('Accuracy for fold %d is: %.2f%%' % ((i + 1), results[i] * 100))

    print('Mean accuracy: %.2f%%' % (results.mean() * 100))
    print('Standard Deviation: %.2f' % results.std())


def main():
    begin_prediction(load_data())


main()
