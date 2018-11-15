import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from decimal import *
import statistics
getcontext().prec = 2


def load_data():
    names = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
             'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv('breast-cancer-wisconsin.data.csv', names=names)
    # print(data[:15])

    imp_mean = SimpleImputer(missing_values='?', strategy='most_frequent')
    data = imp_mean.fit_transform(data)
    data = pd.DataFrame(data=data, index=range(699), columns=names)
    return data


def split_data():
    data = load_data()
    classes = data.iloc[:, 10]
    attributes = data.iloc[:, :10]
    attributes = attributes.astype('int')
    classes = classes.astype('int')
    number_of_instances = len(data)
    num_folds = 5
    fold_size = number_of_instances / num_folds
    accuracies = []
    for i in range(5):
        train_attributes = pd.concat([attributes.iloc[:(int((num_folds - (i + 1)) * fold_size))],
                                      attributes.iloc[int((num_folds - i) * fold_size):]])

        train_classes = pd.concat([classes.iloc[:(int((num_folds - (i + 1)) * fold_size))],
                                   classes.iloc[int((num_folds - i) * fold_size):]])

        test_attributes = attributes.iloc[(int((num_folds - (i + 1)) * fold_size))
                                          :(int((num_folds - i) * fold_size))]

        test_classes = classes.iloc[(int((num_folds - (i + 1)) * fold_size))
                                    :(int((num_folds - i) * fold_size))]
        # print(len(train_attributes))
        print("\t\t\t\tFOLD %d\n" % (i + 1))
        # print(test_classes.head())
        accuracies.append(float(predict(train_attributes, train_classes, test_attributes, test_classes)))
        print('Mean of %a is: %.2f' % (accuracies, statistics.mean(accuracies)))
        print('Standard Deviation is %.2f' % (statistics.pstdev(accuracies)))
    # print(fold_size)


def predict(train_attributes, train_classes, test_attributes, test_classes):
    gnb = KNeighborsClassifier(n_neighbors=2)
    data = load_data()
    y_pred = gnb.fit(train_attributes, train_classes).predict(test_attributes)
    data_size = float(test_attributes.shape[0])
    correctly_predicted = (test_classes != y_pred).sum()
    print("Number of mislabeled points out of a total %d points : %d"
          % (data_size, correctly_predicted))

    #
    inaccuracy = (correctly_predicted / float(data_size)) * 100.
    print("Accuracy is: %.2f" % (100 - inaccuracy))
    print('\n')

    return 100 - inaccuracy


def begin_prediction(data):
    X = data[:, :10]
    Y = data[:, 10]


def main():
    split_data()


main()
