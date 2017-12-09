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
            print ("    True class " + classLbl.split("\\")[-1] + ", predicted class "+ str(predictedClass))
            print("---------------------------------------------------------------")
