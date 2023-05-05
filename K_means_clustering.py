
import pandas as pd
import random
import numpy as np


def getKnnClasses(trn, tst, numberOfNearestNeighbors):
    gN = getNeighbors(tst, trn, numberOfNearestNeighbors)
    predictedClass = talleyNeighbors(gN)
    return predictedClass


def getNeighbors(tstVector, trn, numberOfNearestNeighbors):
    sm = []
    trainDataArray = trn.iloc[:, :-1].to_numpy()
    for i in range(trn.shape[0]):
        sm.append(np.linalg.norm(tstVector - trainDataArray[i, :]))

    sortedIndices = np.argsort(sm)
    knei = trn.iloc[sortedIndices[:numberOfNearestNeighbors], -1]
    return knei.to_list()


def talleyNeighbors(neighbors):
    s, j = np.unique(neighbors, return_inverse=True)
    return s[np.argmax(np.bincount(j))]


# Read the data
data = pd.read_csv('iris.csv', delimiter=',', header=None)

# Get the number of rows in the dataset
numRows = data.shape[0]

# Create a random index for the testing dataset
testIdx = random.sample(range(numRows), 5)

# Create the testing dataset
testData = data.iloc[testIdx, :]

# Create the training dataset by removing the testing dataset rows
trainData = data.drop(testIdx)

# Testing the kNN algorithm
k = 5

accureatePrediction = 0
for i in range(testData.shape[0]):
    testSample = testData.iloc[i, :-1].to_numpy()
    predictedClass = getKnnClasses(trainData, testSample, k)

    print('Row #: ', testIdx[i])
    print('Predicted: ', predictedClass)
    print('Actual: ', testData.iloc[i, -1])

    if predictedClass == testData.iloc[i, -1]:
        print('Prediction is correct.')
        accureatePrediction += 1
    else:
        print('Prediction is incorrect.')
    print('---------------------------')

print('Accuracy of the model: ', (accureatePrediction/5)*100, '%')
