from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from new import calcaulteGMMForEachClass
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def createListWithSpecificNumber(i, maxLen):
    a = []
    a = a + [i] * (maxLen - len(a))
    return a


def getMfccs(fileName):
    (samplingRate, signalSamples) = wav.read(fileName)  # reading the file
    basicMfccfeatures = mfcc(signalSamples,
                             samplingRate)  # this will return (#frames, 13) features which are the 12 basic MFCCs + energy
    deltaMfccFeatures = delta(basicMfccfeatures, 2)  # this will return (#frames, 13) which are the deltas
    doubleDeltaMFCCFeatures = delta(deltaMfccFeatures, 2)  # this will return (#frames, 13) which are the dobule deltas

    allMfccs = np.concatenate((basicMfccfeatures, deltaMfccFeatures, doubleDeltaMFCCFeatures),
                              axis=1)  # this will return (#frames, 39) features
    return allMfccs


# this function will return every class, and the the wav files related to it
# as a dictonary (key is the path of the class, the value is list of files names)
def getTrainingData(folerPath='.\\training_data'):
    rootDir = folerPath

    classes = {}
    for dirName, subdirList, fileList in os.walk(rootDir):
        classes[dirName] = []
        for fileName in fileList:
            classes[dirName].append(fileName)

    classes.pop(rootDir)  # remove the first element (root path)
    return classes

def print_confusion_matrix(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def getTrainingDataMFCCs(folerPath='.\\training_data'):
    classesWithMfcc = {}
    labels = []
    data = []
    classes = {}
    traingDataDictonary = getTrainingData(folerPath)
    for classpath, filesName in traingDataDictonary.items():
        for fileName in filesName:
            filePath = classpath + '\\' + fileName
            # print('reading file %s ...' % filePath)
            currentFileMfccs = getMfccs(filePath)
            # print(currentFileMfccs)
            data.extend(currentFileMfccs)
        classes[classpath] = data
        data = []
    return classes


# this function will return every class, and the the wav files related to it
# as a dictonary (key is the path of the class, the value is list of files names)
def getTestingData(folerPath='.\\testing_data'):
    rootDir = folerPath

    classes = {}
    for dirName, subdirList, fileList in os.walk(rootDir):
        classes[dirName] = []
        for fileName in fileList:
            classes[dirName].append(fileName)

    classes.pop(rootDir)  # remove the first element (root path)
    return classes


def getTestingDataMFCCs(folerPath='.\\testing_data'):
    classesWithMfcc = {}
    traingDataDictonary = getTestingData(folerPath)
    for classpath, filesName in traingDataDictonary.items():
        classesWithMfcc[classpath] = []
        for fileName in filesName:
            filePath = classpath + '\\' + fileName
            print('reading file %s ...' % filePath)
            currentFileMfccs = getMfccs(filePath)
            # print(currentFileMfccs[0, :])  # print first row of mfcc
            classesWithMfcc[classpath].append(currentFileMfccs)
    return classesWithMfcc


if __name__ == '__main__':
    classes = getTrainingDataMFCCs(folerPath='.\\training_data')

    gmms = {}
    count = 0
    for classLabel,classData in classes.items():
        gmms[classLabel] = calcaulteGMMForEachClass(np.array(classData), 1, 10)

    testingDataMfcc = getTestingDataMFCCs(folerPath='.\\testing_data')

    y_true = []
    y_pred = []
    for classLbl, filesMfcc in testingDataMfcc.items():
        for fileMfcc in filesMfcc:
            predictedClassesProbability = []
            predictedClass = 0
            print("Testing file: " + classLbl)
            for label, gmm in gmms.items():
                predictedClassesProbability.append(np.amax(gmm.predict_proba(fileMfcc), axis=0).mean())
                print(" probability for class '" + label.split("\\")[-1] + "' is: " + str(np.amax(gmm.predict_proba(fileMfcc), axis=1).mean()))
            predictedClass = predictedClassesProbability.index(max(predictedClassesProbability))
            print("predicted class is ' " + str(predictedClass) + "'" + " with probability: " + str(predictedClassesProbability[predictedClass]))
            currentTrueClass = classLbl.split("\\")[-1]
            print("    True class " + currentTrueClass + ", predicted class "+ str(predictedClass))
            print("---------------------------------------------------------------")
            y_true.append(int(currentTrueClass))
            y_pred.append(predictedClass)

    #https://gist.github.com/zachguo/10296432
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    print("Concussion Matrix:")
    print_confusion_matrix(confusion_matrix(y_true=y_true, y_pred=y_pred), ['0', '1', '2']) # 0,1,2 are the labels TODO make it generic
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print('precision is : ', precision)
    print('recall is : ', recall)
    print('fscore is : ', fscore)