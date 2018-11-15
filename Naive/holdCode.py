
import csv
import random
import math
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import pandas as pd
import numpy as np
import statistics
np.set_printoptions(precision=2)


def main():
    # cross_validate()
    split_data()


def split_data():
    data = load_data()
    classes = data.iloc[:, 0]
    attributes = data.iloc[:, 1:]
    # print(attributes[:15])
    number_of_instances = len(data)
    num_folds = 5
    fold_size = number_of_instances / num_folds
    accuracies = []
    for i in range(5):
        bound = num_folds - i
        if bound > 4:
            bound = 0
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
        accuracies.append(float(predict(train_attributes, train_classes, test_attributes, test_classes)))
    print('Standard Deviation is %d' % (statistics.pstdev(accuracies)))
    # print(fold_size)


def predict(train_attributes, train_classes, test_attributes, test_classes):
    gnb = GaussianNB()
    data = load_data()
    class_variables = data.iloc[:, 0]
    class_attributes = data.iloc[:, 1:]
    y_pred = gnb.fit(train_attributes, train_classes).predict(test_attributes)
    data_size = float(test_attributes.shape[0])
    correctly_predicted = (test_classes != y_pred).sum()
    print("Number of mislabeled points out of a total %d points : %d"
          % (data_size, correctly_predicted))

    #
    inaccuracy = (correctly_predicted / float(data_size)) * 100.
    print("Accuracy is: %d" % (100 - inaccuracy))
    print('\n')

    return 100 - inaccuracy



def cross_validate():
    gnb = GaussianNB()

    data = load_data()

    # print(data.iloc[0:10, 5:])
    class_variables = data.iloc[:, 0]
    test_variables = pd.concat([data.iloc[0:3, 0], data.iloc[5:7, 0]])
    print("Test Variables: ")
    print(test_variables)
    # print(class_variables[:15])
    class_attributes = data.iloc[:, 1:]
    # print(class_attributes[:15])
    # print(frame[:1000][:1])
    y_pred = gnb.fit(class_attributes[:16000], class_variables[:16000]).predict(class_attributes[16000:])
    data_size = class_attributes[16000:].shape[0] + 1
    correctly_predicted = (class_variables[16000:] != y_pred).sum()
    # print("Number of mislabeled points out of a total %d points : %d"
    #       % (data_size, correctly_predicted))

    inaccuracy = (correctly_predicted // data_size) * 100
    # print("Accuracy is: %d" % (100 - inaccuracy))


def load_data():
    data = pd.read_csv('letter-recognition.data.csv')
    return data


main()
