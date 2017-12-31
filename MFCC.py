from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import numpy as np
from DatasetHelpers import *


    #Generate the MFCCs feature vector from sound file
def getMfccs(fileName):
    # reading the file
    (samplingRate, signalSamples) = wav.read(fileName)

    # this will return (#frames, 13) features which are the 12 basic MFCCs + energy
    basicMfccfeatures = mfcc(signalSamples,
                             samplingRate)

    # this will return (#frames, 13) which are the deltas
    deltaMfccFeatures = delta(basicMfccfeatures, 2)

    # this will return (#frames, 13) which are the dobule deltas
    doubleDeltaMFCCFeatures = delta(deltaMfccFeatures, 2)

    # this will return (#frames, 39) features
    allMfccs = np.concatenate((basicMfccfeatures, deltaMfccFeatures, doubleDeltaMFCCFeatures),
                              axis=1)
    return allMfccs


    #Returns the MFCC feature vector for the training data files.
def getTrainingDataMFCCs(folerPath='.\\training_data'):
    print('Reading training data ...')
    data = []
    classes = {}
    trainingDataDictonary = getTrainingData(folerPath)
    for classpath, filesName in trainingDataDictonary.items():
        for fileName in filesName:
            filePath = classpath + '\\' + fileName
            print('     reading file %s' % filePath)
            currentFileMfccs = getMfccs(filePath)
            data.extend(currentFileMfccs)
        classes[classpath] = data
        data = []

    print('End of training data')
    print('----------------------------------------------------------------------------------------')
    print()
    return classes


    #Returns the MFCC feature vector for the testing data files.
def getTestingDataMFCCs(folerPath='.\\testing_data'):
    print('Reading testing data ...')
    classesWithMfcc = {}
    filesNames = []
    traingDataDictonary = getTestingData(folerPath)
    for classpath, filesName in traingDataDictonary.items():
        classesWithMfcc[classpath] = []
        for fileName in filesName:
            filePath = classpath + '\\' + fileName
            print('     reading file %s ...' % filePath)
            currentFileMfccs = getMfccs(filePath)
            classesWithMfcc[classpath].append(currentFileMfccs)
            filesNames.append(filePath)

    print('End of testing data')
    print('----------------------------------------------------------------------------------------')
    print()
    return filesNames, classesWithMfcc
