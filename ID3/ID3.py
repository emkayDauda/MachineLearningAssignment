import pandas as pd
from sklearn.impute import SimpleImputer


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
    print(data.head())
    return data


def main():
    load_data()


main()
