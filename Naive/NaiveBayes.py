# Example of Naive Bayes implemented from Scratch in Python
from __future__ import division

import csv
import random
import math
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import pandas as pd
import numpy as np

dataSet = []


def loadCsv(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset) - 1):
        restofList = [float(x) for x in dataset[i][1:]]
        restofList.insert(0, dataset[i][0])
        dataset[i] = restofList
    print(dataset[:10])
    dataSet = dataset
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[0] not in separated):
            separated[vector[0]] = []
        separated[vector[0]].append(vector)
    return separated


def mean(numbers):
    print('In Numbers:')
    print(numbers)
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    sk()


def sk():
    gnb = GaussianNB()

    letter = pd.read_csv('letter-recognition.data.csv')

    # print(letter.iloc[0:10, 5:])
    class_variables = letter.iloc[:, 0]
    test_variables = pd.concat([letter.iloc[0:3, 0], letter.iloc[5:7, 0]])
    print("Test Variables: ")
    print(test_variables)
    # print(class_variables[:15])
    class_attributes = letter.iloc[:, 1:]
    # print(class_attributes[:15])
    # print(frame[:1000][:1])
    y_pred = gnb.fit(class_attributes[:16000], class_variables[:16000]).predict(class_attributes[16000:])
    data_size = class_attributes[16000:].shape[0] + 1
    correctly_predicted = (class_variables[16000:] != y_pred).sum()
    # print("Number of mislabeled points out of a total %d points : %d"
    #       % (data_size, correctly_predicted))

    inaccuracy = (correctly_predicted / data_size) * 100
    # print("Accuracy is: %d" % (100 - inaccuracy))


main()
