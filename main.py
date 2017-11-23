from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import scipy.io.wavfile as wav
import os



def getMfccs(fileName):
    (samplingRate, signalSamples) = wav.read(fileName) #reading the file
    basicMfccfeatures = mfcc(signalSamples, samplingRate) # this will return (#frames, 13) features which are the 12 basic MFCCs + energy
    deltaMfccFeatures = delta(basicMfccfeatures, 2) #this will return (#frames, 13) which are the deltas
    doubleDeltaMFCCFeatures = delta(deltaMfccFeatures, 2) #this will return (#frames, 13) which are the dobule deltas

    allMfccs = np.concatenate((basicMfccfeatures, deltaMfccFeatures, doubleDeltaMFCCFeatures), axis=1) #this will return (#frames, 39) features
    return allMfccs


#this function will return every class, and the the wav files related to it
# as a dictonary (key is the path of the class, the value is list of files names)
def getTrainingData(folerPath='.\\training_data'):
    rootDir = folerPath

    classes = {}
    for dirName, subdirList, fileList in os.walk(rootDir):
        classes[dirName]  =[]
        for fileName in fileList:
            classes[dirName].append(fileName)

    classes.pop(rootDir) # remove the first element (root path)
    return classes


def getTrainingDataMFCCs(folerPath='.\\training_data'):
    classesWithMfcc = {}
    traingDataDictonary = getTrainingData(folerPath)
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
    x = getTrainingDataMFCCs(folerPath='.\\training_data')



