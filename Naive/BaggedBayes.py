from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import statistics
np.set_printoptions(precision=2)


def main():
    # cross_validate()
    split_data()


def split_data():
    data = load_data()
    bagged_data = data.sample(n=data.shape[0], replace=True)
    classes = bagged_data.iloc[:, 0]
    attributes = bagged_data.iloc[:, 1:]
    # print(attributes[:15])
    number_of_instances = len(bagged_data)
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
        print("\t\t\t\tFOLD %d\n" % (i + 1))
        accuracies.append(float(predict(train_attributes, train_classes, test_attributes, test_classes)))

    print('Mean of %a is: %.2f' % (accuracies, statistics.mean(accuracies)))
    print('Standard Deviation is %f' % (statistics.pstdev(accuracies)))


def predict(train_attributes, train_classes, test_attributes, test_classes):
    gnb = GaussianNB()
    data = load_data()
    y_pred = gnb.fit(train_attributes, train_classes).predict(test_attributes)
    data_size = float(test_attributes.shape[0])
    correctly_predicted = (test_classes != y_pred).sum()
    print("Number of mislabeled points out of a total %d points : %d"
          % (data_size, correctly_predicted))

    inaccuracy = (correctly_predicted / float(data_size)) * 100.
    print("Accuracy is: %.2f" % (100 - inaccuracy))
    print('\n')

    return 100 - inaccuracy


def load_data():
    data = pd.read_csv('letter-recognition.data.csv')
    return data


main()
