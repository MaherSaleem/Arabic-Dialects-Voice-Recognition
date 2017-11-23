from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import scipy.io.wavfile as wav


if __name__ == '__main__':

    (samplingRate, signalSamples) = wav.read("testSound.wav") #reading the file
    basicMfccfeatures = mfcc(signalSamples, samplingRate) # this will return (#frames, 13) features which are the 12 basic MFCCs + energy
    deltaMfccFeatures = delta(basicMfccfeatures, 2) #this will return (#frames, 13) which are the deltas
    doubleDeltaMFCCFeatures = delta(deltaMfccFeatures, 2) #this will return (#frames, 13) which are the dobule deltas

    allMfccs = np.concatenate((basicMfccfeatures, deltaMfccFeatures, doubleDeltaMFCCFeatures), axis=1) #this will return (#frames, 39) features
    #fbankFeatures = logfbank(signalSamples, samplingRate)


